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

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.common.{ImageProcessing}
import com.intel.analytics.bigdl.transform.vision.image.augmentation

/**
 * adjust the image brightness
 *
 * @param deltaLow brightness parameter: low bound
 * @param deltaHigh brightness parameter: high bound
 */
class Brightness(deltaLow: Double, deltaHigh: Double) extends ImageProcessing {

  private val internalCrop = augmentation.Brightness(deltaLow, deltaHigh)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }
}

object Brightness {
  def apply(deltaLow: Double, deltaHigh: Double): Brightness =
    new Brightness(deltaLow, deltaHigh)
}
