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

package com.intel.analytics.bigdl.models.fasterrcnn.utils

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import com.intel.analytics.bigdl.tensor.Tensor

class Bbox extends Serializable {
  @transient var areasAll: Tensor[Float] = _

  /**
   *
   * @param scoresNms N
   * @param bboxNms   N * 4
   * @param scoresAll M
   * @param bboxAll   M * 4
   * @return
   */
  def bboxVote(scoresNms: Tensor[Float], bboxNms: Tensor[Float],
    scoresAll: Tensor[Float], bboxAll: Tensor[Float]): Target = {
    var accBox: Tensor[Float] = null
    var accScore = 0f
    var box: Tensor[Float] = null
    if (areasAll == null) areasAll = Tensor[Float]
    areasAll.resize(bboxAll.size(1))

    var i = 1
    while (i <= bboxAll.size(1)) {
      areasAll.setValue(i, getArea(bboxAll(i)))
      i += 1
    }
    i = 1
    while (i <= scoresNms.size(1)) {
      box = bboxNms(i)
      if (accBox == null) {
        accBox = Tensor[Float](4)
      } else {
        accBox.fill(0f)
      }
      accScore = 0f
      var m = 1
      while (m <= scoresAll.size(1)) {
        val boxA = bboxAll(m)
        val iw = Math.min(box.valueAt(3), boxA.valueAt(3)) -
          Math.max(box.valueAt(1), boxA.valueAt(1)) + 1
        val ih = Math.min(box.valueAt(4), boxA.valueAt(4)) -
          Math.max(box.valueAt(2), boxA.valueAt(2)) + 1

        if (iw > 0 && ih > 0) {
          val ua = getArea(box) + areasAll.valueAt(m) - iw * ih
          val ov = iw * ih / ua
          if (ov >= 0.5) {
            accBox.add(scoresAll.valueAt(m), boxA)
            accScore += scoresAll.valueAt(m)
          }
        }
        m += 1
      }
      var x = 1
      while (x <= 4) {
        bboxNms.setValue(i, x, accBox.valueAt(x) / accScore)
        x += 1
      }
      i += 1
    }
    Target(scoresNms, bboxNms)
  }

  private def getArea(box: Tensor[Float]): Float = {
    require(box.dim() == 1 && box.nElement() >= 4)
    (box.valueAt(3) - box.valueAt(1) + 1) * (box.valueAt(4) - box.valueAt(2) + 1)
  }

  @transient var deltasCopy: Tensor[Float] = _

  /**
   * Note that the value of boxes and deltas will be changed
   * and the results are saved in boxes
   * @param boxes  (4, N)
   * @param deltas (4, N)
   * @return
   */
  def bboxTransformInv(boxes: Tensor[Float], deltas: Tensor[Float]): Tensor[Float] = {
    if (boxes.size(1) == 0) {
      return Tensor[Float](0, deltas.size(2))
    }
    require(boxes.size(1) == 4 && deltas.size(1) % 4 == 0)
    deltasCopy = if (deltas.isContiguous()) {
      deltas
    } else {
      if (deltasCopy == null) {
        deltasCopy = Tensor[Float]
      }
      deltasCopy.resizeAs(deltas).copy(deltas)
    }

    require(boxes.isContiguous())
    // x2 - x1 + 1
    val widths = boxes(3).add(-1, boxes(1)).add(1f)
    // y2 - y1 + 1
    val heights = boxes(4).add(-1, boxes(2)).add(1f)
    // width / 2 + x1
    val centerX = boxes(1).mul(2).add(widths).div(2)
    // height / 2 + y1
    val centerY = boxes(2).mul(2).add(heights).div(2)

    var ind = 1
    while (ind <= deltasCopy.size(1)) {
      // delta * width + centerX
      deltasCopy(ind).cmul(widths).add(centerX) // predCtrX
      deltasCopy(ind + 1).cmul(heights).add(centerY) // predCtrY
      deltasCopy(ind + 2).exp().cmul(widths).mul(0.5f) // predW
      deltasCopy(ind + 3).exp().cmul(heights).mul(0.5f) // predH
      ind += 4
    }
    ind = 1
    boxes.resizeAs(deltas)
    while (ind <= deltasCopy.size(1)) {
      boxes(ind).add(deltasCopy(ind), -1, deltasCopy(ind + 2))
      boxes(ind + 1).add(deltasCopy(ind + 1), -1, deltasCopy(ind + 3))
      boxes(ind + 2).add(deltasCopy(ind), deltasCopy(ind + 2))
      boxes(ind + 3).add(deltasCopy(ind + 1), deltasCopy(ind + 3))
      ind += 4
    }
    boxes
  }

  @transient var widths: Tensor[Float] = _
  @transient var heights: Tensor[Float] = _

  /**
   * Clip boxes to image boundaries.
   * set the score of all boxes with any side smaller than minSize to 0
   * @param boxes 4a * N
   * @param height
   * @param width
   * @param minH
   * @param minW
   * @param scores
   * @return the number of boxes kept (score > 0)
   */
  def clipBoxes(boxes: Tensor[Float], height: Float, width: Float, minH: Float = 0,
    minW: Float = 0, scores: Tensor[Float] = null): Int = {
    var c = 1
    var r = 1
    var count = 0
    val h = height - 1
    val w = width - 1
    if (scores != null && widths == null) {
      widths = Tensor[Float]
      heights = Tensor[Float]
    }
    while (r <= boxes.size(1)) {
      boxes(r).apply1(x => Math.min(x, w)).cmax(0)
      boxes(r + 1).apply1(x => Math.min(x, h)).cmax(0)
      boxes(r + 2).apply1(x => Math.min(x, w)).cmax(0)
      boxes(r + 3).apply1(x => Math.min(x, h)).cmax(0)
      if (scores != null) {
        widths.resize(boxes.size(2))
        heights.resize(boxes.size(2))
        widths.add(boxes(3), -1, boxes(1)).add(1)
        heights.add(boxes(4), -1, boxes(2)).add(1)
        c = 1
        widths.map(heights, (boxW, boxH) => {
          if (boxW < minW || boxH < minH) scores.setValue(c, 0)
          else count += 1
          c += 1
          boxW
        })
      }
      r += 4
    }
    count
  }
}
