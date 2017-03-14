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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

object GreyImgCropper {
  def apply(cropWidth: Int, cropHeight: Int) : GreyImgCropper = {
    new GreyImgCropper(cropWidth, cropHeight)
  }
}

/**
 * Crop an area from a grey image. The crop area width and height must be smaller than grey image
 * width and height. The area position is random.
 * @param cropWidth
 * @param cropHeight
 */
class GreyImgCropper(cropWidth: Int, cropHeight: Int)
  extends Transformer[LabeledGreyImage, LabeledGreyImage] {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledGreyImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledGreyImage]): Iterator[LabeledGreyImage] = {
    prev.map(img => {
      val width = img.width()
      val height = img.height()
      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
      val startIndex = startW + startH * width
      val frameLength = cropWidth * cropHeight
      val source = img.content
      val target = buffer.content
      var i = 0
      while (i < frameLength) {
        target(i) = source(startIndex + (i / cropWidth) * width +
          (i % cropWidth))
        i += 1
      }

      buffer.setLabel(img.label())
    })
  }
}

