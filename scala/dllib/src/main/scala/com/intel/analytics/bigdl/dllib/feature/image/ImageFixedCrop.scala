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
 * Crop a fixed area of image
 *
 * @param x1 start in width
 * @param y1 start in height
 * @param x2 end in width
 * @param y2 end in height
 * @param normalized whether args are normalized, i.e. in range [0, 1]
 * @param isClip whether to clip the roi to image boundaries
 */
class ImageFixedCrop(x1: Float, y1: Float, x2: Float, y2: Float, normalized: Boolean,
                     isClip: Boolean = true)
  extends ImageProcessing {

  private val internalCrop = new augmentation.FixedCrop(x1, y1, x2, y2, normalized, isClip)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageFixedCrop {
  def apply(x1: Float, y1: Float, x2: Float, y2: Float, normalized: Boolean,
            isClip: Boolean = true)
  : ImageFixedCrop = new ImageFixedCrop(x1, y1, x2, y2, normalized, isClip)
}
