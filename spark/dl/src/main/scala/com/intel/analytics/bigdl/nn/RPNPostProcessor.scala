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

import breeze.linalg.dim
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.{T, Table}


class RPNPostProcessor(preNmsTopNTest: Int,
  postNmsTopNTest: Int,
  nms_thread: Float,
  min_size: Int,
  rpnPostNmsTopNTrain: Int)(
  implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float]{

  @transient private var sortedScores: Tensor[Float] = Tensor[Float]()
  @transient private var sortedInds: Tensor[Float] = Tensor[Float]()
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
    val anchors = input[Tensor[Float]](1).clone() // todo:
    var objectness = input[Tensor[Float]](2).clone() // todo:
    var box_regression = input[Tensor[Float]](3).clone() // todo:
    var imageSize = input[Tensor[Float]](4) // order: height, width

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

    val proposals = BboxUtil.bboxTransformInv(concat_anchors, box_regression_view, normalized = true)
    // remove _small box
    val minBoxH = min_size // * imageSize.valueAt(1)
    val minBoxW = min_size // * imageSize.valueAt(2)
    var keepN = BboxUtil.clipBoxes(proposals, imageSize.valueAt(1), imageSize.valueAt(2), minBoxH
      , minBoxW, sortedScores)

    println(proposals)
    val arr = new Array[Int](1000)
    nms.nms(sortedScores, proposals, thresh = 0.7f, arr, sorted = true)

    val proposals_index = Tensor[Float]()
    val indices = Tensor[Float](T(arr(0), arr(1), arr(2), arr(3), arr(4)))
    output.index(1, indices, proposals)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput = null
    gradInput
  }
}
