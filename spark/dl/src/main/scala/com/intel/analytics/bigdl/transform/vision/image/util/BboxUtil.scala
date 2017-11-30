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

package com.intel.analytics.bigdl.transform.vision.image.util

import com.intel.analytics.bigdl.tensor.Tensor

object BboxUtil {
  def decodeRois(output: Tensor[Float]): Tensor[Float] = {
    // ignore if decoded
    if (output.nElement() < 6 || output.dim() == 2) return output
    val num = output.valueAt(1).toInt
    require(num >= 0)
    if (num == 0) {
      Tensor[Float]()
    } else {
      output.narrow(1, 2, num * 6).view(num, 6)
    }
  }

  // inplace scale
  def scaleBBox(classBboxes: Tensor[Float], height: Float, width: Float): Unit = {
    if (classBboxes.nElement() == 0) return
    classBboxes.select(2, 1).mul(width)
    classBboxes.select(2, 2).mul(height)
    classBboxes.select(2, 3).mul(width)
    classBboxes.select(2, 4).mul(height)
  }

  def locateBBox(srcBox: NormalizedBox, box: NormalizedBox, locBox: NormalizedBox)
  : Unit = {
    val srcW = srcBox.width()
    val srcH = srcBox.height()
    locBox.x1 = srcBox.x1 + box.x1 * srcW
    locBox.y1 = srcBox.y1 + box.y1 * srcH
    locBox.x2 = srcBox.x1 + box.x2 * srcW
    locBox.y2 = srcBox.y1 + box.y2 * srcH
  }

  def jaccardOverlap(bbox: NormalizedBox, bbox2: NormalizedBox): Float = {
    val w = math.min(bbox.x2, bbox2.x2) - math.max(bbox.x1, bbox2.x1)
    if (w < 0) return 0
    val h = math.min(bbox.y2, bbox2.y2) - math.max(bbox.y1, bbox2.y1)
    if (h < 0) return 0
    val overlap = w * h
    overlap / ((bbox.area() + bbox2.area()) - overlap)
  }

  def meetEmitCenterConstraint(srcBox: NormalizedBox, bbox: NormalizedBox): Boolean = {
    val xCenter = bbox.centerX()
    val yCenter = bbox.centerY()
    if (xCenter >= srcBox.x1 && xCenter <= srcBox.x2 &&
      yCenter >= srcBox.y1 && yCenter <= srcBox.y2) {
      true
    } else {
      false
    }
  }

  /**
   * Project bbox onto the coordinate system defined by src_bbox.
   * @param srcBox
   * @param bbox
   * @param projBox
   * @return
   */
  def projectBbox(srcBox: NormalizedBox, bbox: NormalizedBox,
    projBox: NormalizedBox): Boolean = {
    if (bbox.x1 >= srcBox.x2 || bbox.x2 <= srcBox.x1 ||
      bbox.y1 >= srcBox.y2 || bbox.y2 <= srcBox.y1) {
      return false
    }
    val srcWidth = srcBox.width()
    val srcHeight = srcBox.height()
    projBox.x1 = (bbox.x1 - srcBox.x1) / srcWidth
    projBox.y1 = (bbox.y1 - srcBox.y1) / srcHeight
    projBox.x2 = (bbox.x2 - srcBox.x1) / srcWidth
    projBox.y2 = (bbox.y2 - srcBox.y1) / srcHeight
    BboxUtil.clipBox(projBox, projBox)
    if (projBox.area() > 0) true
    else false
  }

  def clipBox(box: NormalizedBox, clipedBox: NormalizedBox): Unit = {
    clipedBox.x1 = Math.max(Math.min(box.x1, 1f), 0f)
    clipedBox.y1 = Math.max(Math.min(box.y1, 1f), 0f)
    clipedBox.x2 = Math.max(Math.min(box.x2, 1f), 0f)
    clipedBox.y2 = Math.max(Math.min(box.y2, 1f), 0f)
  }
}
