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

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{ImageFeature, augmentation}

/**
 * Randomly apply the preprocessing to some of the input ImageFeatures, with probability specified.
 * E.g. if prob = 0.5, the preprocessing will apply to half of the input ImageFeatures.
 *
 * @param preprocessing preprocessing to apply.
 * @param prob probability to apply the preprocessing action
 */
class ImageRandomPreprocessing(
    preprocessing: ImageProcessing,
    prob: Double
  ) extends ImageProcessing  {

  require(prob <= 1.0 && prob >= 0.0, s"prob should be in [0.0, 1.0], now it's $prob")

  private val internalRandomTransformer = new augmentation.RandomTransformer(preprocessing, prob)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRandomTransformer.apply(prev)
  }
}

object ImageRandomPreprocessing {
  def apply(preprocessing: ImageProcessing, prob: Double): ImageRandomPreprocessing =
    new ImageRandomPreprocessing(preprocessing, prob)
}
