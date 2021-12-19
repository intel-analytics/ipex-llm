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
package com.intel.analytics.bigdl.dllib.feature.image

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.augmentation

/**
 * Fill part of image with certain pixel value
 *
 * @param startX start x ratio
 * @param startY start y ratio
 * @param endX end x ratio
 * @param endY end y ratio
 * @param value filling value
 */
class ImageFiller(startX: Float, startY: Float, endX: Float, endY: Float, value: Int = 255)
  extends ImageProcessing {

  private val internalCrop = new augmentation.Filler(startX, startY, endX, endY, value)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageFiller {
  def apply(startX: Float, startY: Float, endX: Float, endY: Float, value: Int = 255): ImageFiller
  = new ImageFiller(startX, startY, endX, endY, value)
}
