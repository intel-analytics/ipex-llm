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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}

import scala.collection.Iterator

object BGRImgToSample {
  def apply(toRGB: Boolean): BGRImgToSample =
    new BGRImgToSample(toRGB)
}

class BGRImgToSample(toRGB: Boolean)
  extends Transformer[LabeledBGRImage, Sample] {
  private val buffer = new Sample()
  private var arrayBuffer: Array[Float] = null

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[Sample] = {
    prev.map(other => {
      val height = other.height
      val width = other.width
      if (arrayBuffer.length < 3 * height * width) {
        arrayBuffer = new Array[Float](3 * height * width)
      }
      other.copyTo(arrayBuffer, 0, toRGB)
      buffer.copy(arrayBuffer, Array(other.label),
        Array(3, height, width), Array(1))
    })
  }
}
