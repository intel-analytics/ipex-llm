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
 * expand image, fill the blank part with the meanR, meanG, meanB
 *
 * @param meansR means in R channel
 * @param meansG means in G channel
 * @param meansB means in B channel
 * @param minExpandRatio min expand ratio
 * @param maxExpandRatio max expand ratio
 */
class ImageExpand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
                  minExpandRatio: Double = 1, maxExpandRatio: Double = 4.0)
  extends ImageProcessing {

  private val internalCrop = new augmentation.Expand(meansR, meansG, meansB,
    minExpandRatio, maxExpandRatio)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageExpand {
  def apply(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
            minExpandRatio: Double = 1.0, maxExpandRatio: Double = 4.0): ImageExpand =
    new ImageExpand(meansR, meansG, meansB, minExpandRatio, maxExpandRatio)
}
