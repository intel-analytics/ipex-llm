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

class NormalizedBox(var x1: Float, var y1: Float, var x2: Float, var y2: Float) {

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

  def this(other: NormalizedBox) {
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

  def clipBox(clipedBox: NormalizedBox): Unit = {
    clipedBox.x1 = Math.max(Math.min(x1, 1f), 0f)
    clipedBox.y1 = Math.max(Math.min(y1, 1f), 0f)
    clipedBox.x2 = Math.max(Math.min(x2, 1f), 0f)
    clipedBox.y2 = Math.max(Math.min(y2, 1f), 0f)
  }

  def scaleBox(height: Float, width: Float, scaledBox: NormalizedBox): Unit = {
    scaledBox.x1 = x1 * width
    scaledBox.y1 = y1 * height
    scaledBox.x2 = x2 * width
    scaledBox.y2 = y2 * height
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case box: NormalizedBox =>
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
}

object NormalizedBox {
  def apply(x1: Float, y1: Float, x2: Float, y2: Float): NormalizedBox =
    new NormalizedBox(x1, y1, x2, y2)

  def apply(box: (Float, Float, Float, Float)): NormalizedBox = {
    new NormalizedBox(box._1, box._2, box._3, box._4)
  }
}
