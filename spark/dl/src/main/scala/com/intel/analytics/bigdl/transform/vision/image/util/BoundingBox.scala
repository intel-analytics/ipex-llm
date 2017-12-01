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

class BoundingBox(var x1: Float, var y1: Float, var x2: Float, var y2: Float,
  var normalized: Boolean = true) extends Serializable {

  var label: Float = -1

  def setLabel(l: Float): this.type = {
    label = l
    this
  }

  var difficult: Float = -1

  def setDifficult(d: Float): this.type = {
    difficult = d
    this
  }

  def this(other: BoundingBox) {
    this(other.x1, other.y1, other.x2, other.y2)
  }

  def centerX(): Float = {
    (x1 + x2) / 2
  }

  def centerY(): Float = {
    (y1 + y2) / 2
  }

  def this() = {
    this(0f, 0f, 1f, 1f)
  }

  def width(): Float = x2 - x1

  def height(): Float = y2 - y1

  def area(): Float = width() * height()

  def clipBox(clipedBox: BoundingBox): Unit = {
    if (normalized) {
      clipedBox.x1 = Math.max(Math.min(x1, 1f), 0f)
      clipedBox.y1 = Math.max(Math.min(y1, 1f), 0f)
      clipedBox.x2 = Math.max(Math.min(x2, 1f), 0f)
      clipedBox.y2 = Math.max(Math.min(y2, 1f), 0f)
    }
  }

  def scaleBox(height: Float, width: Float, scaledBox: BoundingBox): Unit = {
    scaledBox.x1 = x1 * width
    scaledBox.y1 = y1 * height
    scaledBox.x2 = x2 * width
    scaledBox.y2 = y2 * height
  }

  /**
   * Whether the center of given bbox lies in current bbox
   */
  def meetEmitCenterConstraint(bbox: BoundingBox): Boolean = {
    val xCenter = bbox.centerX()
    val yCenter = bbox.centerY()
    if (xCenter >= x1 && xCenter <= x2 &&
      yCenter >= y1 && yCenter <= y2) {
      true
    } else {
      false
    }
  }

  /**
   * whether overlaps with given bbox
   */
  def isOverlap(bbox: BoundingBox): Boolean = {
    if (bbox.x1 >= x2 || bbox.x2 <= x1 || bbox.y1 >= y2 || bbox.y2 <= y1) {
      false
    } else {
      true
    }
  }

  def jaccardOverlap(bbox: BoundingBox): Float = {
    val w = math.min(x2, bbox.x2) - math.max(x1, bbox.x1)
    if (w < 0) return 0
    val h = math.min(y2, bbox.y2) - math.max(y1, bbox.y1)
    if (h < 0) return 0
    val overlap = w * h
    overlap / ((area() + bbox.area()) - overlap)
  }

  /**
   * Project bbox onto the coordinate system defined by current bbox.
   * @param bbox
   * @param projBox
   * @return
   */
  def projectBbox(bbox: BoundingBox, projBox: BoundingBox): Boolean = {
    if (!isOverlap(bbox)) {
      return false
    }
    val srcWidth = width()
    val srcHeight = height()
    projBox.x1 = (bbox.x1 - x1) / srcWidth
    projBox.y1 = (bbox.y1 - y1) / srcHeight
    projBox.x2 = (bbox.x2 - x1) / srcWidth
    projBox.y2 = (bbox.y2 - y1) / srcHeight
    projBox.clipBox(projBox)
    if (projBox.area() > 0) true else false
  }

  def locateBBox(box: BoundingBox, locBox: BoundingBox)
  : Unit = {
    val srcW = width()
    val srcH = height()
    locBox.x1 = x1 + box.x1 * srcW
    locBox.y1 = y1 + box.y1 * srcH
    locBox.x2 = x1 + box.x2 * srcW
    locBox.y2 = y1 + box.y2 * srcH
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case box: BoundingBox =>
        if (box.x1 == x1 && box.x2 == x2 && box.y1 == y1 && box.y2 == y2) true
        else false
      case _ => false
    }
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + x1.hashCode()
    hash = hash * seed + y1.hashCode()
    hash = hash * seed + x2.hashCode()
    hash = hash * seed + y2.hashCode()

    hash
  }

  override def toString: String = {
    s"BoundingBox ($x1, $y1, $x2, $y2)"
  }
}

object BoundingBox {
  def apply(x1: Float, y1: Float, x2: Float, y2: Float,
    normalized: Boolean = true): BoundingBox =
    new BoundingBox(x1, y1, x2, y2, normalized)
}
