/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.fasterrcnn

import com.intel.analytics.bigdl.nn.Nms
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

/**
 * Outputs object detection proposals by applying estimated bounding-box
 * transformations to a set of regular boxes (called "anchors").
 * rois: holds R regions of interest, each is a 5-tuple
 * (n, x1, y1, x2, y2) specifying an image batch index n and a rectangle (x1, y1, x2, y2)
 * scores: holds scores for R regions of interest
 *
 */
class Proposal(preNmsTopN: Int, postNmsTopN: Int, anchorParam: AnchorParam)
  extends AbstractModule[Table, Table, Float] {

  private val anchorUtil: Anchor = new Anchor(anchorParam)
  @transient private var nms: Nms = _
  @transient private var bboxDeltas: Tensor[Float] = _
  @transient private var scores: Tensor[Float] = _
  @transient private var keep: Array[Int] = _
  @transient private var rpnRois: Tensor[Float] = _
  @transient private var sortedScores: Tensor[Float] = _
  @transient private var sortedInds: Tensor[Float] = _
  @transient private var filteredProposals: Tensor[Float] = _
  // Proposal height and width both need to be greater than minSize (at orig image scale)
  private val minSize = 16

  private def init(): Unit = {
    if (nms == null) {
      nms = Nms()
      bboxDeltas = Tensor[Float]
      scores = Tensor[Float]
      sortedScores = Tensor[Float]
      sortedInds = Tensor[Float]
      filteredProposals = Tensor[Float]
      rpnRois = Tensor[Float]
    }
  }

  /**
   * Algorithm:
   * for each (H, W) location i
   * generate A anchor boxes centered on cell i
   * apply predicted bbox deltas at cell i to each of the A anchors
   * clip predicted boxes to image
   * remove predicted boxes with either height or width < threshold
   * sort all (proposal, score) pairs by score from highest to lowest
   * take top pre_nms_topN proposals before NMS
   * apply NMS with threshold to remaining proposals
   * take after_nms_topN proposals after NMS
   * return the top proposals (-> RoIs top, scores top)
   * @param input input(1): cls scores
   *              input(2): bbox pred
   *              input(3): im_info
   * @return output
   *         output(1): rpn_rois
   *         output(2): rpn_scores
   */
  override def updateOutput(input: Table): Table = {
    val inputScore = input[Tensor[Float]](1)
    require(inputScore.size(1) == 1, "currently only support single batch")
    init()
    // transpose from (1, 4A, H, W) to (H * W * A, 4)
    transposeAndReshape(input[Tensor[Float]](2), 4, bboxDeltas)

    // select scores for object (while the remaining is the score for background)
    // transpose from (1, 2A, H, W) to (H * W * A)
    val scoresOri = inputScore.narrow(2, anchorParam.num + 1, anchorParam.num)
    transposeAndReshape(scoresOri, 1, scores)

    val imInfo = input[Tensor[Float]](3)

    // Generate proposals from bbox deltas and shifted anchors
    // Enumerate all shifts
    val anchors = anchorUtil.generateAnchors(inputScore.size(4), inputScore.size(3), 16)
    // Convert anchors into proposals via bbox transformations
    val proposals = BboxUtil.bboxTransformInv(anchors, bboxDeltas)
    // clip predicted boxes to image
    // original faster rcnn way
    // minimum box width & height
    val minBoxH = minSize * imInfo.valueAt(3)
    val minBoxW = minSize * imInfo.valueAt(4)
    var keepN = BboxUtil.clipBoxes(proposals, imInfo.valueAt(1), imInfo.valueAt(2), minBoxH
      , minBoxW, scores)

    val topNum = Math.min(preNmsTopN, keepN)
    scores.topk(topNum, dim = 1, increase = false,
      result = sortedScores, indices = sortedInds)
    if (keep == null || keep.length < sortedInds.nElement()) {
      keep = new Array[Int](sortedInds.nElement())
    }
    var k = 1
    while (k <= sortedInds.nElement()) {
      keep(k - 1) = sortedInds.valueAt(k).toInt - 1
      k += 1
    }
    filteredProposals.resize(topNum, proposals.size(2))
    k = 1
    while (k <= topNum) {
      filteredProposals.update(k, proposals(keep(k - 1) + 1))
      k += 1
    }

    // apply nms (e.g. threshold = 0.7)
    // take after_nms_topN (e.g. 300)
    // return the top proposals (-> RoIs topN
    keepN = nms.nms(sortedScores, filteredProposals, 0.7f, keep, sorted = true)
    if (postNmsTopN > 0) {
      keepN = Math.min(keepN, postNmsTopN)
    }

    var i = 1
    var j = 2
    rpnRois.resize(keepN, filteredProposals.size(2) + 1)
    while (i <= keepN) {
      rpnRois.setValue(i, 1, 0)
      j = 2
      while (j <= rpnRois.size(2)) {
        rpnRois.setValue(i, j, filteredProposals.valueAt(keep(i - 1), j - 1))
        j += 1
      }
      i += 1
    }
    if (output.length == 0) {
      output.insert(rpnRois)
    } else {
      output.update(1, rpnRois)
    }
    output
  }

  // Transpose and reshape predicted bbox transformations to get them
  // into the same order as the anchors:
  // bbox deltas will be (1, 4 * A, H, W) format
  // transpose to (1, H, W, 4 * A)
  // reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
  // in slowest to fastest order
  private def transposeAndReshape(mat: Tensor[Float], cols: Int,
    out: Tensor[Float]): Tensor[Float] = {
    if (cols == 1) {
      out.resize(mat.nElement())
    } else {
      out.resize(mat.nElement() / cols, cols)
    }
    val matArr = mat.storage().array()
    val matOffset = mat.storageOffset() - 1
    val st2 = mat.stride(2)
    val st3 = mat.stride(3)
    val outArr = out.storage().array()
    var outOffset = out.storageOffset() - 1
    var ind = 0
    var r = 0
    while (r < mat.size(3)) {
      var c = 0
      val offset3 = r * st3
      while (c < mat.size(4)) {
        var i = 0
        while (i < mat.size(2)) {
          var j = 0
          while (j < cols) {
            outArr(outOffset) = matArr(matOffset + (i + j) * st2 + offset3 + c)
            outOffset += 1
            j += 1
          }
          i += cols
          ind += 1
        }
        c += 1
      }
      r += 1
    }
    out
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput
    gradInput
  }

  override def toString: String = "Proposal"
}
