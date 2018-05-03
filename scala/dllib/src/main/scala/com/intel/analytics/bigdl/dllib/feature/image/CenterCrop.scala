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

import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, augmentation}
import com.intel.analytics.zoo.feature.common.Preprocessing
class CenterCrop(
    cropWidth: Int,
    cropHeight: Int,
    isClip: Boolean = true) extends Preprocessing[ImageFeature, ImageFeature] {

  private val internalCrop = augmentation.CenterCrop(cropWidth, cropHeight)
  def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }
}

object CenterCrop {
  def apply(cropWidth: Int, cropHeight: Int, isClip: Boolean = true)
  : CenterCrop = new CenterCrop(cropWidth, cropHeight, isClip)
}

