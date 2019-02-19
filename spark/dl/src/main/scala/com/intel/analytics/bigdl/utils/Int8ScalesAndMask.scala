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

package com.intel.analytics.bigdl.utils

import scala.collection.mutable.ArrayBuffer

class Int8ScalesAndMask(val mask: Int) extends Serializable {

  private val _scales: ArrayBuffer[Array[Float]] = ArrayBuffer.empty[Array[Float]]

  def scales: ArrayBuffer[Array[Float]] = _scales

  def set(scales: Array[Array[Float]]): Unit = {
    _scales.clear()
    scales.foreach(append)
  }

  def get(): Array[Array[Float]] = {
    _scales.toArray
  }

  def getMask() : Int = mask

  def append(scale: Array[Float]): Unit = {
    _scales.append(scale)
  }

  def update(scale: Array[Float], index: Int): Unit = {
    if (scales.length - 1 < index) {
      scales.append(scale)
    }

    scales(index).indices.foreach(i =>
      if (scale(i) > scales(index)(i)) {
        scales(index)(i) = scale(i)
      })
  }
}

object Int8ScalesAndMask {
  val SINGLE_SCALE = 0
}
