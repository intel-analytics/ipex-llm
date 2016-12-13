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

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object RGBImgCropper {
  def apply(cropWidth: Int, cropHeight: Int): RGBImgCropper =
    new RGBImgCropper(cropWidth, cropHeight)
}

class RGBImgCropper(cropWidth: Int, cropHeight: Int)
  extends Transformer[LabeledRGBImage, LabeledRGBImage] {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledRGBImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledRGBImage]): Iterator[LabeledRGBImage] = {
    prev.map(img => {
      val width = img.width()
      val height = img.height()
      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
      val startIndex = (startW + startH * width) * 3
      val frameLength = cropWidth * cropHeight
      val source = img.content
      val target = buffer.content
      var i = 0
      while (i < frameLength) {
        target(i * 3 + 2) =
          source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3 + 2)
        target(i * 3 + 1) =
          source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3 + 1)
        target(i * 3) =
          source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3)
        i += 1
      }
      buffer.setLabel(img.label())
    })
  }
}
