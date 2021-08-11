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

/**
 * Image crop method, e.g. random, center
 */
sealed trait CropperMethod

/**
 * crop the random position of image
 */
case object CropRandom extends CropperMethod

/**
 * crop the center of image
 */
case object CropCenter extends CropperMethod

object BGRImgCropper {
  def apply(cropWidth: Int, cropHeight: Int,
    cropperMethod: CropperMethod = CropRandom): BGRImgCropper =
    new BGRImgCropper(cropWidth, cropHeight, cropperMethod)
}

/**
 * Crop a `cropWidth` x `cropHeight` patch from an image. The patch size should be less than
 * the image size. There're two cropping methods: at random and from the center. The former
 * is preferred for simple data augmentation during training while the later applies to
 * validation or testing
 * @param cropWidth
 * @param cropHeight
 * @param cropperMethod
 */
class BGRImgCropper(cropWidth: Int, cropHeight: Int, cropperMethod: CropperMethod = CropRandom)
  extends Transformer[LabeledBGRImage, LabeledBGRImage] {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new LabeledBGRImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledBGRImage] = {
    prev.map(img => {
      val width = img.width()
      val height = img.height()
      val (startH, startW) = cropperMethod match {
        case CropRandom =>
          (math.ceil(RNG.uniform(1e-2, height - cropHeight)).toInt,
            math.ceil(RNG.uniform(1e-2, width - cropWidth)).toInt)
        case CropCenter =>
          ((height - cropHeight) / 2, (width - cropWidth) / 2)
      }
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
