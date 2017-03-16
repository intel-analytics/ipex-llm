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

package com.intel.analytics.bigdl.dataset.roiimage

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

/**
 * Image with region of interest
 * @param data   pixel data
 * @param imInfo (heightAfterScale, widthAfterScale, scaleh, scalew)
 * @param target image label, optinal
 */
class RoiImage(
  var data: Array[Float],
  val imInfo: Tensor[Float],
  var target: Option[Target] = None) extends Serializable {

  var path: String = ""

  def this() = this(new Array[Float](0), Tensor[Float](4))

  // height after scale
  def height: Int = imInfo.valueAt(1).toInt

  // width after scale
  def width: Int = imInfo.valueAt(2).toInt

  // scale ratio for height
  def scaledH: Float = imInfo.valueAt(3)

  // scale ratio for weidth
  def scaledW: Float = imInfo.valueAt(4)

  def copy(rawData: Array[Byte]): this.type = {
    require(rawData.length == 8 + height * width * 3)
    if (data.length < height * width * 3) {
      data = new Array[Float](height * width * 3)
    }
    var i = 0
    while (i < width * height * 3) {
      data(i) = rawData(i + 8) & 0xff
      i += 1
    }
    this
  }

  def copyTo(storage: Array[Float], offset: Int, toRGB: Boolean = true): Unit = {
    val frameLength = width * height
    require(frameLength * 3 + offset <= storage.length)
    var j = 0
    if (toRGB) {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3 + 2)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3 + 2)
        j += 1
      }
    }
  }
}

/**
 * Image with byte data and target (optional), path
 * @param data byte array, this is transformed from java BufferedImage
 * @param dataLength length of byte array
 * @param target label, optional
 * @param path image path
 */
case class RoiByteImage(data: Array[Byte], var dataLength: Int,
  var target: Option[Target] = None, path: String)


/**
 * Image path and target information
 * @param imagePath image path
 * @param target image target
 * @param flipped whether this image is flipped
 */
case class RoiImagePath(
  imagePath: String,
  target: Option[Target] = None,
  flipped: Boolean = false) {
}

/**
 * image target with classes and bounding boxes
 * @param classes N
 * @param bboxes 4N
 */
case class Target(classes: Tensor[Float], bboxes: Tensor[Float]) {
  def toTable: Table = {
    val table = new Table()
    table.insert(classes)
    table.insert(bboxes)
  }
}

