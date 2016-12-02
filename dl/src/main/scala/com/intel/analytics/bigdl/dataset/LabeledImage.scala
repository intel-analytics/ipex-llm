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

class LabeledImage(
  private var data: Array[Float],
  private var _width: Int,
  private var _height: Int,
  private var _nChannels: Int,
  private var _label: Float = 0
) extends Serializable {

  def this(width: Int, height: Int, numChannels: Int) =
    this(new Array[Float](width * height * numChannels),
      width, height, numChannels)

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
  def update(row: Int, col: Int, channelIdx: Int, newVal: Float): this.type = {
    data(imageToVectorCoords(row, col, channelIdx)) = newVal
    this
  }

  def content: Array[Float] = data

  def label: Float = _label

  def setLabel(label: Float): this.type = {
    this._label = label
    this
  }

  def width: Int = _width

  def height: Int = _height

  def nChannels: Int = _nChannels

  def length: Int = _nChannels * _height * _width

  @inline def imageToVectorCoords(row: Int, col: Int, c: Int): Int = {
    (row * width + col) * nChannels + c
  }

  def copy(rawData: Array[Byte], numChannels: Int, normalize: Float = 255.0f): this.type = {
    val buffer = ByteBuffer.wrap(rawData)
    _width = buffer.getInt
    _height = buffer.getInt
    require(rawData.length == 8 + width * height * numChannels)
    if (data.length < height * width * numChannels) {
      data = new Array[Float](width * height * numChannels)
    }
    var i = 0
    while (i < height * width * numChannels) {
      data(i) = (rawData(i + 8) & 0xff) / normalize
      i += 1
    }
    this
  }

  def copyTo(storage: Array[Float], offset: Int): Unit = {
    require(length + offset <= storage.length)
    var i = 0
    while (i < nChannels) {
      var j = 0
      while (j < width * height) {
        storage(offset + j + width * height * i) = content(j * nChannels + i)
        j += 1
      }
      i +=  1
    }
  }

  def copy(other: LabeledImage): LabeledImage = {
    _height = other.height
    _width = other.width
    _nChannels = other.nChannels
    this._label = other.label
    if (this.data.length < length) {
      this.data = new Array[Float](length)
    }

    var i = 0
    while (i < length) {
      this.data(i) = other.data(i)
      i += 1
    }
    this
  }
}
