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

package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator

sealed trait CropperMethod

case object CropRandom extends CropperMethod

case object CropCenter extends CropperMethod

object BGRImgCropper {
  def apply(cropWidth: Int,
            cropHeight: Int,
            cropperMethod: CropperMethod = CropRandom,
            padding: Int = 0
           ): BGRImgCropper =
    new BGRImgCropper(cropWidth, cropHeight, cropperMethod, padding)
}

/**
 * Crop a `cropWidth` x `cropHeight` patch from an image. The patch size should not exceed the
 * original image size (when the `padding` option is enabled, crop **after** padding). There're two
 * cropping methods: at random and from the center. The former is preferred for simple data
 * augmentation during training while the later applies to validation or testing.
 *
 * @param cropWidth width of the patch
 * @param cropHeight height of the patch
 * @param cropperMethod crop at random and from the center are supported
 * @param padding by default there is no padding
 */
class BGRImgCropper(cropWidth: Int, cropHeight: Int, cropperMethod: CropperMethod = CropRandom,
                    padding: Int = 0) extends Transformer[LabeledBGRImage, LabeledBGRImage] {
  require(padding >= 0, "padding size must be non-negative")
  require(cropWidth > 0, "cropWidth must be positive")
  require(cropHeight > 0, "cropHeight must be positive")

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledBGRImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      val width = img.width() + 2 * padding
      val height = img.height() + 2 * padding
      require(cropWidth <=  width, "crop width is out of range")
      require(cropHeight <= height, "crop height is out of range")

      val (startH, startW) = cropperMethod match {
        case CropRandom =>
          (RNG.uniform(0, height - cropHeight).toInt, RNG.uniform(0, width - cropWidth).toInt)
        case CropCenter =>
          ((height - cropHeight) / 2, (width - cropWidth) / 2)
      }
      val frameLength = cropWidth * cropHeight
      val source = img.content
      val target = buffer.content
      var i = 0
      while (i < frameLength) {
        val h_i = i / cropWidth + startH
        val w_i = i % cropWidth + startW

        if (h_i >= padding && h_i < img.height() + padding &&
          w_i >= padding && w_i < img.width() + padding) {
          val offset = (h_i - padding) * img.width() + (w_i - padding)
          target(i * 3 + 2) = source(offset * 3 + 2)
          target(i * 3 + 1) = source(offset * 3 + 1)
          target(i * 3) = source(offset * 3)
        } else { // fells to the padding positions
          target(i * 3 + 2) = 0
          target(i * 3 + 1) = 0
          target(i * 3) = 0
        }
        i += 1
      }
      buffer.setLabel(img.label())
    })
  }
}
