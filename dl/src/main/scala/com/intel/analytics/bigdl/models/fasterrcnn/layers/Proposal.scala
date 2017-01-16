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

package com.intel.analytics.bigdl.models.fasterrcnn.layers

import com.intel.analytics.bigdl.models.fasterrcnn.utils._
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

  @transient var anchorTool: Anchor = _
  @transient var nmsTool: Nms = _
  @transient var bboxTool: Bbox = _
  @transient var bboxDeltas: Tensor[Float] = _
  @transient var scores: Tensor[Float] = _
  @transient var filteredScores: Tensor[Float] = _
  @transient var keep: Array[Int] = _
  @transient var keepN: Int = 0
  @transient var rpnRois: Tensor[Float] = _
  @transient var sortedScores: Tensor[Float] = _
  @transient var sortedInds: Tensor[Float] = _
  @transient var filteredProposals: Tensor[Float] = _
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
  val minSize = 16

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
    if (anchorTool == null) anchorTool = new Anchor(anchorParam)
    if (nmsTool == null) nmsTool = new Nms
    if (bboxTool == null) bboxTool = new Bbox

    // the first set of _num_anchors channels are bg probs
    // the second set are the fg probs, which we want
    val dataSize = input[Tensor[Float]](1).size()

    // Transpose and reshape predicted bbox transformations to get them
    // into the same order as the anchors:
    // bbox deltas will be (1, 4 * A, H, W) format
    // transpose to (1, H, W, 4 * A)
    // reshape to (4, 1 * H * W * A) where rows are ordered by (h, w, a)
    // in slowest to fastest order
    def transposeAndReshape(mat: Tensor[Float], cols: Int, out: Tensor[Float]): Tensor[Float] = {
      out.resize(cols, mat.nElement() / cols)
      var ind = 1
      var r = 1
      var c = 1
      var i = 1
      var j = 1
      while (r <= mat.size(3)) {
        c = 1
        while (c <= mat.size(4)) {
          i = 1
          while (i <= mat.size(2)) {
            j = 1
            while (j <= cols) {
              out.setValue(j, ind, mat.valueAt(1, i + j - 1, r, c))
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

    // bbox_deltas: (1, 4A, H, W)
    if (bboxDeltas == null) bboxDeltas = Tensor[Float]
    transposeAndReshape(input[Tensor[Float]](2), 4, bboxDeltas)

    if (scores == null) scores = Tensor[Float]
    val scoresOri = input[Tensor[Float]](1).narrow(2, anchorParam.num + 1, anchorParam.num)
    transposeAndReshape(scoresOri, 1, scores).squeeze()

    val imInfo = input[Tensor[Float]](3)

    // Generate proposals from bbox deltas and shifted anchors
    // Enumerate all shifts
    val anchors = anchorTool.getAllAnchors(dataSize(3), dataSize(2), 16)

    // Convert anchors into proposals via bbox transformations
    val proposals = bboxTool.bboxTransformInv(anchors, bboxDeltas)
    // clip predicted boxes to image
    // original faster rcnn way
    // minimum box width & height
    val minBoxH = minSize * imInfo.valueAt(3)
    val minBoxW = minSize * imInfo.valueAt(4)
    keepN = bboxTool.clipBoxes(proposals, imInfo.valueAt(1), imInfo.valueAt(2), minBoxH
      , minBoxW, scores)

    val topNum = Math.min(preNmsTopN, keepN)
    if (sortedScores == null) {
      sortedScores = Tensor[Float]
      sortedInds = Tensor[Float]
    }
    scores.topk(topNum, dim = 1, increase = false,
      result = sortedScores, indices = sortedInds)
    if (keep == null || keep.length < sortedInds.nElement()) {
      keep = new Array[Int](sortedInds.nElement())
    }
    var k = 1
    while (k <= sortedInds.nElement()) {
      keep(k - 1) = sortedInds.valueAt(k).toInt
      k += 1
    }
    if (filteredProposals == null) filteredProposals = Tensor[Float]
    TensorUtil.selectMatrix(proposals, keep, 2, topNum, filteredProposals)
    // apply nms (e.g. threshold = 0.7)
    // take after_nms_topN (e.g. 300)
    // return the top proposals (-> RoIs top)
    keepN = nmsTool.nms(sortedScores, filteredProposals, 0.7f, keep)
    if (postNmsTopN > 0) {
      keepN = Math.min(keepN, postNmsTopN)
    }

    if (rpnRois == null) rpnRois = Tensor[Float]
    var i = 1
    var j = 2
    rpnRois.resize(keepN, filteredProposals.size(1) + 1)
    while (i <= keepN) {
      rpnRois.setValue(i, 1, 0)
      j = 2
      while (j <= rpnRois.size(2)) {
        rpnRois.setValue(i, j, filteredProposals.valueAt(j - 1, keep(i - 1)))
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

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput
    gradInput
  }

  override def toString: String = "layers.Proposal"
}

case class ProposalParam()
