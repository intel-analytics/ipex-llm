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

/**
 * image channel normalize
 */
class ChannelNormalizer(
    means: Array[Float],
    stds: Array[Float]
  ) extends Preprocessing[ImageFeature, ImageFeature] {

  private val internalCrop = new augmentation.ChannelNormalize(means, stds)
  def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }
}

object ChannelNormalizer {
  /**
   * image channel normalize
   *
   * @param meanR mean value in R channel
   * @param meanG mean value in G channel
   * @param meanB mean value in B channel
   * @param stdR  std value in R channel
   * @param stdG  std value in G channel
   * @param stdB  std value in B channel
   */
  def apply(meanR: Float, meanG: Float, meanB: Float,
            stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): ChannelNormalizer = {
    new ChannelNormalizer(Array(meanB, meanG, meanR), Array(stdR, stdG, stdB))
  }

  def apply(mean: Float, std: Float): ChannelNormalizer = {
    new ChannelNormalizer(Array(mean), Array(std))
  }
}
