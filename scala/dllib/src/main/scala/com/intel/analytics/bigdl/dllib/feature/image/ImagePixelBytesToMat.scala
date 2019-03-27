/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, PixelBytesToMat, augmentation}
import org.apache.log4j.Logger

/**
 * Transform byte array(pixels in byte) to OpenCVMat
 *
 * @param byteKey key that maps byte array
 */
class ImagePixelBytesToMat(
      byteKey: String = ImageFeature.bytes) extends ImageProcessing {

  private val internalTransformer = new PixelBytesToMat(byteKey)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalTransformer.apply(prev)
  }
}

object ImagePixelBytesToMat {
  val logger = Logger.getLogger(getClass)
  def apply(byteKey: String = ImageFeature.bytes): ImagePixelBytesToMat = {
    new ImagePixelBytesToMat(byteKey)
  }
}
