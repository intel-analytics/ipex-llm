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
import breeze.linalg.dim
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.Convolution2D
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.{T, Table}
import com.sun.tracing.dtrace.ModuleName
import org.apache.spark.api.java.function

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
 * @param min_size
 * @param rpnPostNmsTopNTrain
 */
class RegionRroposal[T: ClassTag](inChannels: Int,
   anchorSizes: Array[Float],
   aspectRatios: Array[Float],
   anchorStride: Array[Float],
   preNmsTopNTest: Int,
   postNmsTopNTest: Int,
   nmsThread: Float,
   min_size: Int,
   rpnPostNmsTopNTrain: Int)(implicit ev: TensorNumeric[T])
   extends AbstractModule[Table, Tensor[T], T] {
  // for anchor generate
  require(anchorSizes.length == anchorStride.length, s"anchor size and stride should be same")
  private val scalesForStride = new Array[Float](1)
  private val anchors = new ArrayBuffer[Anchor]
  for (i <- 0 to anchorSizes.length - 1) {
    scalesForStride(0) = anchorSizes(i) / anchorStride(i)
    anchors.append(Anchor(aspectRatios, scalesForStride))
  }
  private[nn] def anchorGenerater(featuresMap: Tensor[T]): Table = {
    val res = T()
    val length = Math.min(anchorSizes.length, featuresMap.size(1))
    for (i <- 0 to length - 1) {
      val size = anchorSizes(i)
      val stride = anchorStride(i)
      val feature = featuresMap.select(1, i + 1)
      val height = feature.size(2)
      val width = feature.size(3)
      res(i + 1) = anchors(i).generateAnchors(width, height, stride)
    }
    res
  }

  private val numAnchors = anchors(0).anchorNum
  private val head = new RPNHead(inChannels, numAnchors)
  private val boxSelector = new ProposalPostProcessor(preNmsTopNTest, postNmsTopNTest,
    nmsThread, min_size, rpnPostNmsTopNTrain)

  /**
   * input is a table and contains:
   * first tensor: images: images for which we want to compute the predictions
   * second tensor: features: features computed from the images that are used for
   *  computing the predictions.
   */
  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 2 && !this.isTraining(), "Only support tests")
    val images = input[Tensor[T]](1)
    val features = input[Tensor[T]](2)

    val anchors = this.anchorGenerater(features)
    val headOutput = head.forward(features).toTable
    val objectness = headOutput.apply[Tensor[T]](1)
    val rpn_box_regression = headOutput.apply[Tensor[T]](2)

    output = boxSelector.forward(T(anchors[Tensor[T]](1), objectness,
      rpn_box_regression, images))
    output.toTensor[T]
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = null
    gradInput
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    head.parameters()
  }

  override def getParametersTable(): Table = {
    head.getParametersTable()
  }
}

private[bigdl] class ProposalPostProcessor[T: ClassTag](preNmsTopNTest: Int,
  postNmsTopNTest: Int,
  nms_thread: Float,
  min_size: Int,
  rpnPostNmsTopNTrain: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float]{

  @transient private val sortedScores: Tensor[Float] = Tensor[Float]()
  @transient private val sortedInds: Tensor[Float] = Tensor[Float]()
  @transient private val sigmoid = Sigmoid[Float]()
  @transient private val nms = new Nms()

  /**
    * Arguments:
    *    anchors: list[BoxList] -> should be Tensor with shape (N, nums, 4)
    *    objectness: tensor of size N, A, H, W
    *    box_regression: tensor of size N, A * 4, H, W
    *    img_info: image size
    * @param input
    * @return
    */
  override def updateOutput(input: Table): Tensor[Float] = {
    //  for memory case, input may be changed
    val anchors = input[Tensor[Float]](1)
    var objectness = input[Tensor[Float]](2)
    var box_regression = input[Tensor[Float]](3)
    val imageSize = input[Tensor[Float]](4) // order: height, width

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
    val topNum = Math.min(preNmsTopNTest, num_anchors)
    // scores ==> objectness
    // sortedScores ===> objectness, sortedInds = topk_idx + 1
    objectness.topk(topNum, dim = 2, increase = false,
      result = sortedScores, indices = sortedInds)

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
    val minBoxH = min_size
    val minBoxW = min_size
    var keepN = BboxUtil.clipBoxes(proposals, imageSize.valueAt(1), imageSize.valueAt(2), minBoxH
      , minBoxW, sortedScores)

    val arr = new Array[Int](1000)
    nms.nms(sortedScores, proposals, thresh = nms_thread, arr, sorted = true)
    val arrFilter = arr.filter(_ > 0).map(_.toFloat)

    val indices = Tensor[Float]().set(Storage(arrFilter), 1, Array(arrFilter.length))
    output.index(1, indices, proposals)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput = null
    gradInput
  }
}

