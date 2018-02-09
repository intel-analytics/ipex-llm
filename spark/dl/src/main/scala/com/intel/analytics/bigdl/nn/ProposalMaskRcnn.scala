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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.Table

class ProposalMaskRcnn(preNmsTopNTest: Int, postNmsTopNTest: Int,
  rpnPreNmsTopNTrain: Int, rpnPostNmsTopNTrain: Int)(
  implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

  @transient private var nms: Nms = _
  @transient private var bboxDeltas: Tensor[Float] = _
  @transient private var scores: Tensor[Float] = _
  @transient private var sortedScores: Tensor[Float] = _
  @transient private var sortedInds: Tensor[Float] = _
  @transient private var filteredDetas: Tensor[Float] = _
  @transient private var filteredAnchors: Tensor[Float] = _
  @transient private var keep: Array[Int] = _

  private def init(): Unit = {
    if (nms == null) {
      nms = new Nms()
      bboxDeltas = Tensor[Float]
      scores = Tensor[Float]
      sortedScores = Tensor[Float]
      sortedInds = Tensor[Float]
      filteredDetas = Tensor[Float]
      filteredAnchors = Tensor[Float]
    }
  }

  override def updateOutput(input: Table): Tensor[Float] = {
    val inputScore = input[Tensor[Float]](1)
    val rpnBboxes = input[Tensor[Float]](2)
    init()
    require(rpnBboxes.dim() == 3)
    bboxDeltas.resizeAs(rpnBboxes).copy(rpnBboxes)

    // do it due to pretrained order is y1, x1, y2, x2, change to x1, y1, x2, y2
    bboxDeltas.narrow(3, 1, 1).copy(rpnBboxes.narrow(3, 2, 1))
    bboxDeltas.narrow(3, 2, 1).copy(rpnBboxes.narrow(3, 1, 1))
    bboxDeltas.narrow(3, 3, 1).copy(rpnBboxes.narrow(3, 4, 1))
    bboxDeltas.narrow(3, 4, 1).copy(rpnBboxes.narrow(3, 3, 1))

    // remove the batch dim
    bboxDeltas.squeeze(1)
    bboxDeltas.narrow(2, 1, 2).mul(0.1f)
    bboxDeltas.narrow(2, 3, 2).mul(0.2f)

    val fgScores = inputScore.narrow(3, 2, 1)
    scores.resizeAs(fgScores).copy(fgScores)
    scores.resize(inputScore.size(2))

    val priorBoxes = input[Tensor[Float]](3)
    val anchors = priorBoxes.reshape(Array(priorBoxes.size(2), priorBoxes.size(3)))

    val preNmsTopN = if (isTraining()) rpnPreNmsTopNTrain else preNmsTopNTest
    val postNmsTopN = if (isTraining()) rpnPostNmsTopNTrain else postNmsTopNTest

    // Improve performance by trimming to top anchors by score
    // and doing the rest on the smaller subset.
    val preNmsLimit = Math.min(preNmsTopN, anchors.size(1))
    scores.topk(preNmsLimit, dim = 1, increase = false,
      result = sortedScores, indices = sortedInds)

    BboxUtil.selectTensor(bboxDeltas, sortedInds.storage().array().map(_.toInt),
      1, out = filteredDetas)

    BboxUtil.selectTensor(anchors, sortedInds.storage().array().map(_.toInt),
      1, out = filteredAnchors)

    // Apply deltas to anchors to get refined anchors.
    // [batch, N, (y1, x1, y2, x2)]
    // config it
    val height = ProposalMaskRcnn.height
    val width = ProposalMaskRcnn.width
    // Convert anchors into proposals via bbox transformations
    val boxes = BboxUtil.bboxTransformInv(filteredAnchors, filteredDetas, true)
    // Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    BboxUtil.clipBoxes(boxes, height, width)

    // Normalize dimensions to range of 0 to 1.
    BboxUtil.scaleBBox(boxes, 1 / height, 1 / width)

    if (keep == null || keep.length < sortedInds.nElement()) {
      keep = new Array[Int](sortedInds.nElement())
    }
    // apply nms (e.g. threshold = 0.7)
    // take after_nms_topN (e.g. 300)
    // return the top proposals (-> RoIs topN
    var keepN = nms.nmsFast(sortedScores, boxes, 0.7f, 0, keep)
    if (postNmsTopN > 0) {
      keepN = Math.min(keepN, postNmsTopN)
    }

    var i = 1
    var j = 2

    output.resize(keepN, boxes.size(2))
    while (i <= keepN) {
      output.setValue(i, 1, 0)
      j = 1
      while (j <= output.size(2)) {
        output.setValue(i, j, boxes.valueAt(keep(i - 1), j))
        j += 1
      }
      i += 1
    }
    output.resize(1, output.size(1), output.size(2))

    output
  }


  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput = null
    gradInput
  }
}

object ProposalMaskRcnn {
  def apply(preNmsTopNTest: Int, postNmsTopNTest: Int,
    rpnPreNmsTopNTrain: Int = 12000, rpnPostNmsTopNTrain: Int = 2000)(
    implicit ev: TensorNumeric[Float]): ProposalMaskRcnn =
    new ProposalMaskRcnn(preNmsTopNTest, postNmsTopNTest,
      rpnPreNmsTopNTrain, rpnPostNmsTopNTrain)

  val height = 1024f
  val width = 1024f
}
