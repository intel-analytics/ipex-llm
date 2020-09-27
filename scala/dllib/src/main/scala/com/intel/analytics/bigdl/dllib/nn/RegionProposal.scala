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

import java.util
import breeze.linalg.{dim, min}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.{LayerException, T, Table}
import scala.collection.mutable.ArrayBuffer

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
class RegionProposal(
   val inChannels: Int,
   val anchorSizes: Array[Float],
   val aspectRatios: Array[Float],
   val anchorStride: Array[Float],
   val preNmsTopNTest: Int = 1000,
   val postNmsTopNTest: Int = 1000,
   val preNmsTopNTrain: Int = 2000,
   val postNmsTopNTrain: Int = 2000,
   val nmsThread: Float = 0.7f,
   val minSize: Int = 0)(implicit ev: TensorNumeric[Float])
   extends AbstractModule[Table, Table, Float] {

  // for anchor generation
  require(anchorSizes.length == anchorStride.length,
      s"length of anchor size and stride should be same, " +
      s"but get size length ${anchorSizes.length}, stride length ${anchorStride.length}")

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
   * Add a simple RPN Head with classification and regression heads
   */
  private[nn] def rpnHead(inChannels: Int, numAnchors: Int): Module[Float] = {
    val conv = SpatialConvolution[Float](inChannels, inChannels,
      kernelH = 3, kernelW = 3, strideH = 1, strideW = 1, padH = 1, padW = 1)
    conv.setInitMethod(RandomNormal(0.0, 0.01), Zeros)
    val clsLogits = SpatialConvolution[Float](inChannels, numAnchors,
      kernelH = 1, kernelW = 1, strideH = 1, strideW = 1).setName(this.getName() + "cls_logits")
    clsLogits.setInitMethod(RandomNormal(0.0, 0.01), Zeros)
    val bboxPred = SpatialConvolution[Float](inChannels, numAnchors * 4,
      kernelH = 1, kernelW = 1, strideH = 1, strideW = 1).setName(this.getName() + "bbox_pred")
    bboxPred.setInitMethod(RandomNormal(0.0, 0.01), Zeros)

    val input = Input()
    val node1 = conv.inputs(input)
    val node2 = ReLU[Float]().inputs(node1)
    val node3 = clsLogits.inputs(node2)
    val node4 = bboxPred.inputs(node2)

    Graph(input, Array(node3, node4))
  }

  /**
   * input is a table and contains:
   * first tensor: features: features computed from the images that are used for
   * computing the predictions.
   * second tensor: image height and image width
   */
  override def updateOutput(input: Table): Table = {
    val features = input[Table](1)
    val imageSize = input[Tensor[Float]](2)
    val anchors = this.anchorGenerator(features)

    // for batch
    val batchSize = features[Tensor[Float]](1).size(1)
    for (b <- 1 to batchSize) {
      var bboxNumber = 0
      var i = 1
      while (i <= anchors.length()) {
        val singleFeatures = features[Tensor[Float]](i).narrow(1, b, 1)
        val headOutput = head.forward(singleFeatures).toTable
        val objectness = headOutput.apply[Tensor[Float]](1)
        val boxRegression = headOutput.apply[Tensor[Float]](2)

        val out = boxSelector.forward(T(anchors[Tensor[Float]](i), objectness,
          boxRegression, imageSize))

        if (!selectorRes.contains(i)) selectorRes(i) = T(Tensor[Float](), Tensor[Float]())
        selectorRes(i).asInstanceOf[Table].apply[Tensor[Float]](1).resizeAs(out[Tensor[Float]](1))
          .copy(out[Tensor[Float]](1))
        selectorRes(i).asInstanceOf[Table].apply[Tensor[Float]](2).resizeAs(out[Tensor[Float]](2))
          .copy(out[Tensor[Float]](2))

        bboxNumber += selectorRes[Table](i)[Tensor[Float]](1).size(1)
        i += 1
      }

      val postNmsTopN = if (this.isTraining()) min(postNmsTopNTrain, bboxNumber)
      else min(postNmsTopNTest, bboxNumber)

      if (!output.contains(b)) {
        output(b) = Tensor[Float]()
      }
      output[Tensor[Float]](b).resize(postNmsTopN, 4)

      // sort
      selectOverAllLevels(selectorRes, postNmsTopN, bboxNumber, output[Tensor[Float]](b))
    }
    // clear others tensors in output
    for (i <- (batchSize + 1) to output.length()) {
     output.remove[Tensor[Float]](i)
    }
    output
  }

  /**
   * different behavior during training and during testing:
   * during training, post_nms_top_n is over *all* the proposals combined, while
   * during testing, it is over the proposals for each image
   */
  private def selectOverAllLevels(res: Table, postNmsTopN: Int, totalNumber: Int,
                                  output: Tensor[Float]): Unit = {
    val scoreResult = Tensor[Float]().resize(totalNumber)
    val bboxResult = Tensor[Float]().resize(totalNumber, 4)
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

    val inds = scoreResult.topk(postNmsTopN, dim = 1, sortedResult = true, increase = false)

    i = 1
    while (i <= inds._2.nElement()) {
      val index = inds._2.valueAt(i).toInt
      output.narrow(1, i, 1).copy(bboxResult.narrow(1, index, 1))
      i += 1
    }
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    throw new UnsupportedOperationException("RegionProposal only support inference")
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    throw new UnsupportedOperationException("RegionProposal only support inference")
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

  override def training(): RegionProposal.this.type = {
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

object RegionProposal {
  def apply(inChannels: Int,
            anchorSizes: Array[Float] = Array[Float](32, 64, 128, 256, 512),
            aspectRatios: Array[Float] = Array[Float](0.5f, 1.0f, 2.0f),
            anchorStride: Array[Float] = Array[Float](4, 8, 16, 32, 64),
            preNmsTopNTest: Int = 1000,
            postNmsTopNTest: Int = 1000,
            preNmsTopNTrain: Int = 2000,
            postNmsTopNTrain: Int = 2000,
            nmsThread: Float = 0.7f,
            minSize: Int = 0)(implicit ev: TensorNumeric[Float]): RegionProposal =
    new RegionProposal(inChannels, anchorSizes, aspectRatios, anchorStride,
      preNmsTopNTest, postNmsTopNTest, preNmsTopNTrain, postNmsTopNTrain, nmsThread,
      minSize)
}

private[nn] class ProposalPostProcessor(
  val preNmsTopNTest: Int = 1000,
  val postNmsTopNTest: Int = 1000,
  val preNmsTopNTrain: Int = 2000,
  val postNmsTopNTrain: Int = 2000,
  val nmsThread: Float = 0.7f,
  val minSize: Int = 0)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Table, Float]{

  @transient private var sortedScores: Tensor[Float] = null
  @transient private var sortedInds: Tensor[Float] = null
  @transient private var boxRegressionIndex: Tensor[Float] = null
  @transient private var anchorsIndex: Tensor[Float] = null

  private val nms = new Nms()
  private val arr = new Array[Int](10000)
  private val sigmoid = Sigmoid[Float]()

  /**
   * Arguments:
   *    anchors: Tensor with shape (batchsize, nums, 4)
   *    objectness: Tensor of size (batchsize, anchornumber, height, width)
   *    box_regression: Tensor of size (batchsize, anchornumber * 4, height, width)
   *    img_info: image size
   * @param input
   * @return
   */
  override def updateOutput(input: Table): Table = {
    //  for memory case, input may be changed
    val anchors = input[Tensor[Float]](1)
    var objectness = input[Tensor[Float]](2)
    var boxRegression = input[Tensor[Float]](3)
    val imageSize = input[Tensor[Float]](4) // image height & width

    val N = objectness.size(1) // batch size
    val A = objectness.size(2) // anchor number
    val H = objectness.size(3) // height
    val W = objectness.size(4) // width

    // permute_and_flatten
    objectness = objectness.transpose(3, 1).transpose(2, 4).contiguous()
    // view[N, -1]
    objectness.resize(Array(N, A * H * W))
    // sigmoid
    objectness = sigmoid.forward(objectness)

    // permute_and_flatten
    boxRegression = boxRegression.transpose(3, 1).transpose(2, 4)
      .contiguous().resize(Array(N, A * H * W, 4))

    val numAnchors = A * H * W
    val topNum = if (this.isTraining()) {
      Math.min(preNmsTopNTrain, numAnchors)
    } else Math.min(preNmsTopNTest, numAnchors)
    // scores ==> objectness
    // sortedScores ===> objectness, sortedInds = topk_idx + 1
    // initial
    if (sortedScores == null) sortedScores = Tensor[Float]()
    if (sortedInds == null) sortedInds = Tensor[Float]()

    objectness.topk(topNum, dim = 2, increase = false,
      result = sortedScores, indices = sortedInds)

    objectness.resizeAs(sortedScores).copy(sortedScores)

    if (boxRegressionIndex == null) boxRegressionIndex = Tensor[Float]()
    boxRegressionIndex.resizeAs(boxRegression)
    boxRegressionIndex.index(2, sortedInds.squeeze(1), boxRegression)
    // view (-1, 4)
    boxRegressionIndex.resize(boxRegressionIndex.nElement() / 4, 4)

    // view (-1, 4)
    if (anchorsIndex == null) anchorsIndex = Tensor[Float]()
    anchorsIndex.resizeAs(anchors)
    anchorsIndex.index(1, sortedInds.squeeze(1), anchors)
    anchorsIndex.resize(anchorsIndex.nElement() / 4, 4)

    val proposals = BboxUtil.bboxTransformInv(anchorsIndex,
      boxRegressionIndex, normalized = true)
    // remove _small box and clip to images
    val minBoxH = minSize
    val minBoxW = minSize
    var keepN = BboxUtil.clipBoxes(proposals, imageSize.valueAt(1), imageSize.valueAt(2), minBoxH
      , minBoxW, sortedScores)

    util.Arrays.fill(arr, 0, arr.length, 0)
    nms.nms(sortedScores, proposals, thresh = nmsThread, arr, sorted = true, orderWithBBox = false)
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


