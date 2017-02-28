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

import com.intel.analytics.bigdl.tensor.Tensor

class BboxUtil extends Serializable {

  /**
   * Note that the output are stored in input boxes
   * and the results are saved in boxes
   * @param boxes  (N, 4)
   * @param deltas (N, 4)
   * @return
   */
  def bboxTransformInv(boxes: Tensor[Float], deltas: Tensor[Float]): Tensor[Float] = {
    if (boxes.size(1) == 0) {
      return boxes
    }
    require(boxes.size(2) == 4 && deltas.size(2) == 4)
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    val deltasArr = deltas.storage().array()
    var deltasoffset = deltas.storageOffset() - 1
    var i = 0
    while (i < boxes.size(1)) {
      val x1 = boxesArr(offset)
      val y1 = boxesArr(offset + 1)
      val width = boxesArr(offset + 2) - x1 + 1
      val height = boxesArr(offset + 3) - y1 + 1
      val predCtrX = deltasArr(deltasoffset) * width + x1 + width / 2 // dx1*width + centerX
      val predCtrY = deltasArr(deltasoffset + 1) * height + y1 + height / 2 // dy1*height + centerY
      val predW = Math.exp(deltasArr(deltasoffset + 2)).toFloat * width / 2 // exp(dx2)*width/2
      val predH = Math.exp(deltasArr(deltasoffset + 3)).toFloat * height / 2 // exp(dy2)*height/2
      boxesArr(offset) = predCtrX - predW
      boxesArr(offset + 1) = predCtrY - predH
      boxesArr(offset + 2) = predCtrX + predW
      boxesArr(offset + 3) = predCtrY + predH
      offset += rowLength
      deltasoffset += rowLength
      i += 1
    }
    boxes
  }

  /**
   * Clip boxes to image boundaries.
   * set the score of all boxes with any side smaller than minSize to 0
   * @param boxes  N * 4
   * @param height height of image
   * @param width  width of image
   * @param minH   min height limit
   * @param minW   min width limit
   * @param scores scores for boxes
   * @return the number of boxes kept (score > 0)
   */
  def clipBoxes(boxes: Tensor[Float], height: Float, width: Float, minH: Float,
    minW: Float, scores: Tensor[Float]): Int = {
    require(boxes.size(2) == 4, "boxes should have the shape N*4")
    require(boxes.size(1) == scores.size(1), "the rows of boxes should be" +
      " equal to the score elements")
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    val scoresArr = scores.storage().array()
    val scoreOffset = scores.storageOffset() - 1
    var c = 1
    var r = 0
    var count = 0
    val h = height - 1
    val w = width - 1
    require(boxes.size(2) == 4)
    while (r < boxes.size(1)) {
      boxesArr(offset) = Math.max(Math.min(boxesArr(offset), w), 0)
      boxesArr(offset + 1) = Math.max(Math.min(boxesArr(offset + 1), h), 0)
      boxesArr(offset + 2) = Math.max(Math.min(boxesArr(offset + 2), w), 0)
      boxesArr(offset + 3) = Math.max(Math.min(boxesArr(offset + 3), h), 0)
      val width = boxesArr(offset + 2) - boxesArr(offset) + 1
      if (width < minW) {
        scoresArr(scoreOffset + r) = 0
      } else {
        val height = boxesArr(offset + 3) - boxesArr(offset + 1) + 1
        if (height < minH) scoresArr(scoreOffset + r) = 0
        else count += 1
      }
      offset += rowLength
      r += 1
    }
    count
  }
}

object BboxUtil {
  def apply(): BboxUtil = new BboxUtil()
}
