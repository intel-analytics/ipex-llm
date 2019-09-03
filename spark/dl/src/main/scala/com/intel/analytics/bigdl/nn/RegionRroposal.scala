/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import breeze.linalg
import breeze.linalg.{dim, min}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.Convolution2D
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{LayerException, T, Table}
import com.sun.tracing.dtrace.ModuleName
import org.apache.spark.api.java.function
import org.dmg.pmml.True
import sun.misc.GC

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Layer for RPN computation. Takes feature maps from the backbone and
 * outputs RPN proposals and losses.
 * @param inChannels
 * @param anchorSizes
 * @param aspectRatios
 * @param anchorStride
 * @param preNmsTopNTest
 * @param postNmsTopNTest
 * @param nmsThread
 * @param minSize
 */
class RegionRroposal(
   val inChannels: Int,
   val anchorSizes: Array[Float],
   val aspectRatios: Array[Float],
   val anchorStride: Array[Float],
   val preNmsTopNTest: Int = 1000,
   val postNmsTopNTest: Int = 1000,
   val preNmsTopNTrain: Int = 2000,
   val postNmsTopNTrain: Int = 2000,
   val nmsThread: Float = 0.7f,
   val minSize: Int = 0,
   val fpnPostNmsTopN: Int = 2000)(implicit ev: TensorNumeric[Float])
   extends AbstractModule[Table, Tensor[Float], Float] {

  // for anchor generation
  require(anchorSizes.length == anchorStride.length, s"anchor size and stride should be same")

  private val scalesForStride = new Array[Float](1)
  private val anchors = new ArrayBuffer[Anchor]
  for (i <- 0 to anchorSizes.length - 1) {
    scalesForStride(0) = anchorSizes(i) / anchorStride(i)
    anchors.append(Anchor(aspectRatios, scalesForStride))
  }

  private val numAnchors = anchors(0).anchorNum
  private val head = rpnHead(inChannels, numAnchors)
  private val boxSelector = new ProposalPostProcessor(preNmsTopNTest, postNmsTopNTest,
    preNmsTopNTrain, postNmsTopNTrain, nmsThread, minSize)
  private val selectorRes = T()

  private[nn] def anchorGenerator(featuresMap: Table): Table = {
    val res = T()
    val length = Math.min(anchorSizes.length, featuresMap.length())
    for (i <- 0 to length - 1) {
      val size = anchorSizes(i)
      val stride = anchorStride(i)
      val feature = featuresMap[Tensor[Float]](i + 1)
      val height = feature.size(3)
      val width = feature.size(4)
      res(i + 1) = anchors(i).generateAnchors(width, height, stride)
    }
    res
  }

  /**
   * Adds a simple RPN Head with classification and regression heads
   */
  private[nn] def rpnHead(inChannels: Int, numAnchors: Int): Module[Float] = {
    val conv = SpatialConvolution[Float](inChannels, inChannels,
      kernelH = 3, kernelW = 3, strideH = 1, strideW = 1, padH = 1, padW = 1)
    conv.setInitMethod(RandomNormal(0.0, 0.01), Zeros)
    val conv2 = SpatialConvolution[Float](inChannels, numAnchors,
      kernelH = 1, kernelW = 1, strideH = 1, strideW = 1).setName(this.getName() + "_cls_logits")
    conv2.setInitMethod(RandomNormal(0.0, 0.01), Zeros)
    val conv3 = SpatialConvolution[Float](inChannels, numAnchors * 4,
      kernelH = 1, kernelW = 1, strideH = 1, strideW = 1).setName(this.getName() + "_bbox_pred")
    conv3.setInitMethod(RandomNormal(0.0, 0.01), Zeros)

    val input = Input()
    val node1 = conv.inputs(input)
    val node2 = ReLU[Float]().inputs(node1)
    val node3 = conv2.inputs(node2)
    val node4 = conv3.inputs(node2)

    Graph(input, Array(node3, node4))
  }

  /**
   * input is a table and contains:
   * first tensor: images: images for which we want to compute the predictions
   * second tensor: features: features computed from the images that are used for
   *  computing the predictions.
   */
  override def updateOutput(input: Table): Tensor[Float] = {
    require(!this.isTraining(), "Only support RegionProposal inference")
    val features = input[Table](1)
    val images = input[Tensor[Float]](2)
    val anchors = this.anchorGenerator(features)

    var bboxNumber = 0
    var i = 1
    while (i <= anchors.length()) {
      val headOutput = head.forward(features(i)).toTable
      val objectness = headOutput.apply[Tensor[Float]](1)
      val rpn_box_regression = headOutput.apply[Tensor[Float]](2)

      val out = boxSelector.forward(T(anchors[Tensor[Float]](i), objectness,
        rpn_box_regression, images)).clone()

      if (!selectorRes.contains(i)) selectorRes(i) = T(Tensor[Float](), Tensor[Float]())
      selectorRes(i).asInstanceOf[Table].apply[Tensor[Float]](1).resizeAs(out[Tensor[Float]](1))
        .copy(out[Tensor[Float]](1))
      selectorRes(i).asInstanceOf[Table].apply[Tensor[Float]](2).resizeAs(out[Tensor[Float]](2))
        .copy(out[Tensor[Float]](2))

      bboxNumber += selectorRes[Table](i)[Tensor[Float]](1).size(1)
      i += 1
    }

    output.resize(bboxNumber, 4)

    val post_nms_top_n = min(fpnPostNmsTopN, bboxNumber)
    // sort
    selectOverAllLevels(selectorRes, post_nms_top_n, output)
    output
  }

  private def selectOverAllLevels(res: Table, post_nms_top_n: Int, output: Tensor[Float]): Unit = {
    val scoreResult = Tensor[Float]().resize(post_nms_top_n)
    val bboxResult = Tensor[Float]().resize(post_nms_top_n, 4)
    var i = 1
    var startOffset = 1
    while (i <= res.length()) {
      val tmpScore = res[Table](i)[Tensor[Float]](2)
      val tmpBbox = res[Table](i)[Tensor[Float]](1)
      scoreResult.narrow(1, startOffset, tmpScore.size(1)).copy(tmpScore)
      bboxResult.narrow(1, startOffset, tmpBbox.size(1)).copy(tmpBbox)
      startOffset = startOffset + tmpScore.size(1)
      i += 1
    }

    val inds = scoreResult.topk(post_nms_top_n, dim = 1, sortedResult = true, increase = false)

    i = 1
    while (i <= inds._2.nElement()) {
      val index = inds._2.valueAt(i).toInt
      output.narrow(1, i, 1).copy(bboxResult.narrow(1, index, 1))
      i += 1
    }
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    throw new UnsupportedOperationException("RegionRroposal only support inference")
  }

  override def accGradParameters(input: Table, gradOutput: Tensor[Float]): Unit = {
    throw new UnsupportedOperationException("RegionRroposal only support inference")
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    head.parameters()
  }

  override def getParametersTable(): Table = {
    head.getParametersTable()
  }

  override def clearState(): this.type = {
    super.clearState()
    head.clearState()
    boxSelector.clearState()
    this
  }

  override def release(): Unit = {
    super.release()
    head.release()
    boxSelector.release()
  }

  override def training(): RegionRroposal.this.type = {
    train = true
    head.training()
    boxSelector.training()
    super.training()
  }

  override def evaluate(): this.type = {
    head.evaluate()
    boxSelector.evaluate()
    train = false
    super.evaluate()
  }
}

object RegionRroposal {
  def apply(inChannels: Int,
            anchorSizes: Array[Float] = Array[Float](32, 64, 128, 256, 512),
            aspectRatios: Array[Float] = Array[Float](0.5f, 1.0f, 2.0f),
            anchorStride: Array[Float] = Array[Float](4, 8, 16, 32, 64),
            preNmsTopNTest: Int = 1000,
            postNmsTopNTest: Int = 1000,
            preNmsTopNTrain: Int = 2000,
            postNmsTopNTrain: Int = 2000,
            nmsThread: Float = 0.7f,
            minSize: Int = 0,
            fpnPostNmsTopN: Int = 2000)(implicit ev: TensorNumeric[Float]): RegionRroposal =
    new RegionRroposal(inChannels, anchorSizes, aspectRatios, anchorStride,
      preNmsTopNTest, postNmsTopNTest, preNmsTopNTrain, postNmsTopNTrain, nmsThread,
      minSize, fpnPostNmsTopN)
}

private[nn] class ProposalPostProcessor(
  val preNmsTopNTest: Int = 1000,
  val postNmsTopNTest: Int = 1000,
  val preNmsTopNTrain: Int = 2000,
  val postNmsTopNTrain: Int = 2000,
  val nmsThread: Float = 0.7f,
  val minSize: Int = 0)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float]{

  @transient private val sortedScores: Tensor[Float] = Tensor[Float]()
  @transient private val sortedInds: Tensor[Float] = Tensor[Float]()
  @transient private val sigmoid = Sigmoid[Float]()
  @transient private val nms = new Nms()

  /**
   * Arguments:
   *    anchors: Tensor with shape (N, nums, 4)
   *    objectness: Tensor of size N, A, H, W
   *    box_regression: Tensor of size N, A * 4, H, W
   *    img_info: image size
   * @param input
   * @return
   */
  override def updateOutput(input: Table): Table = {
    //  for memory case, input may be changed
    val anchors = input[Tensor[Float]](1)
    var objectness = input[Tensor[Float]](2)
    var box_regression = input[Tensor[Float]](3)
    val imageSize = input[Tensor[Float]](4) // height, width

    val N = objectness.size(1)
    val A = objectness.size(2)
    val H = objectness.size(3)
    val W = objectness.size(4)

    // permute_and_flatten
    objectness = objectness.transpose(3, 1).transpose(2, 4).contiguous()
    // view[N, -1]
    objectness.resize(Array(N, A * H * W))
    // sigmoid
    objectness = sigmoid.forward(objectness)

    // permute_and_flatten
    box_regression = box_regression.transpose(3, 1).transpose(2, 4)
      .contiguous().resize(Array(N, A * H * W, 4))

    val num_anchors = A * H * W
    val topNum = if (this.isTraining()) {
      Math.min(preNmsTopNTest, num_anchors)
    } else Math.min(preNmsTopNTrain, num_anchors)
    // scores ==> objectness
    // sortedScores ===> objectness, sortedInds = topk_idx + 1
    objectness.topk(topNum, dim = 2, increase = false,
      result = sortedScores, indices = sortedInds)

    objectness.resizeAs(sortedScores).copy(sortedScores)

    val tmp = Tensor[Float]().resizeAs(box_regression)
    tmp.index(2, sortedInds.squeeze(1), box_regression)

    // view (-1, 4)
    val tmp_anchors = Tensor[Float]().resizeAs(anchors)
    tmp_anchors.index(1, sortedInds.squeeze(1), anchors)
    val concat_anchors = tmp_anchors.resize(tmp_anchors.nElement() / 4, 4)
    // view (-1, 4)
    val box_regression_view = tmp.resize(tmp.nElement() / 4, 4)

    val proposals = BboxUtil.bboxTransformInv(concat_anchors,
      box_regression_view, normalized = true)
    // remove _small box
    val minBoxH = minSize
    val minBoxW = minSize
    var keepN = BboxUtil.clipBoxes(proposals, imageSize.valueAt(1), imageSize.valueAt(2), minBoxH
      , minBoxW, sortedScores)

    val arr = new Array[Int](100000)
    nms.nms(sortedScores, proposals, thresh = nmsThread, arr, sorted = true)
    val arrFilter = arr.filter(_ > 0).map(_.toFloat)

    val indices = Tensor[Float]().set(Storage(arrFilter), 1, Array(arrFilter.length))

    // initial output tensors
    if (output.length() == 0) {
      output(1) = Tensor[Float]()
      output(2) = Tensor[Float]()
    }
    output[Tensor[Float]](1).resize(indices.nElement(), 4).zero()
    output[Tensor[Float]](2).resize(indices.nElement()).zero()

    output[Tensor[Float]](1).index(1, indices, proposals)
    objectness.resize(objectness.nElement())
    output[Tensor[Float]](2).index(1, indices, objectness)

    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    throw new UnsupportedOperationException("ProposalPostProcessor only support inference")
  }
}


