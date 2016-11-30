/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.dataset

import java.nio.ByteBuffer

case class ImageMetadata(var width: Int, var height: Int, var numChannels: Int)

class Image(
  var data: Array[Float],
  var metadata: ImageMetadata,
  var label: Float = 0
) extends Serializable {

  def this(width: Int, height: Int, numChannels: Int) =
    this(new Array[Float](width * height * numChannels),
      new ImageMetadata(width, height, numChannels))

  /**
   * Get the pixel value at (row, col, channelIdx).  Channels are indexed as
   * follows:
   * - If the image is Grey, 0 => gray pixels
   * - If the image is RGB, 0 => blue, 1 => green, 2 => red
   */
  def get(row: Int, col: Int, channelIdx: Int): Float = {
    data(imageToVectorCoords(row, col, channelIdx))
  }

  /**
   * Put a pixel value at (row, col, channelIdx).
   */
  def put(row: Int, col: Int, channelIdx: Int, newVal: Float): this.type = {
    data(imageToVectorCoords(row, col, channelIdx)) = newVal
    this
  }

  def content: Array[Float] = data

  def getLabel: Float = label

  def setLabel(label: Float): this.type = {
    this.label = label
    this
  }

  def width: Int = metadata.width

  def height: Int = metadata.height

  def numChannels: Int = metadata.numChannels

  def getLength: Int = metadata.numChannels * metadata.height * metadata.width

  def imageToVectorCoords(row: Int, col: Int, c: Int): Int = {
    (row * width + col) * numChannels + c
  }

  def copy(rawData: Array[Byte], numChannels: Int, scale: Float = 255.0f): this.type = {
    val buffer = ByteBuffer.wrap(rawData)
    metadata.width = buffer.getInt
    metadata.height = buffer.getInt
    require(rawData.length == 8 + width * height * numChannels)
    if (data.length < height * width * numChannels) {
      data = new Array[Float](width * height * numChannels)
    }
    var i = 0
    while (i < height * width * numChannels) {
      data(i) = (rawData(i + 8) & 0xff) / scale
      i += 1
    }
    this
  }

  def copyTo(storage: Array[Float], offset: Int): Unit = {
    require(getLength + offset <= storage.length)
    var i = 0
    while (i < numChannels) {
      var j = 0
      while (j < width * height) {
        storage(offset + j + width * height * i) = content(j * numChannels + i)
        j += 1
      }
      i +=  1
    }
  }

  def copy(other: Image): Image = {
    metadata.height = other.height
    metadata.width = other.width
    metadata.numChannels = other.numChannels
    this.label = other.label
    if (this.data.length < getLength) {
      this.data = new Array[Float](getLength)
    }

    var i = 0
    while (i < getLength) {
      this.data(i) = other.data(i)
      i += 1
    }
    this
  }
}
