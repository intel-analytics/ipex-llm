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

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{ImageFeature, augmentation}
import org.opencv.core.Core

/**
 * image channel normalize
 */
class ImageChannelNormalize(
    means: Array[Float],
    stds: Array[Float]) extends ImageProcessing {

  private val internalCrop = new augmentation.ChannelNormalize(means, stds)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageChannelNormalize {
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
            stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): ImageChannelNormalize = {
    new ImageChannelNormalize(Array(meanB, meanG, meanR), Array(stdB, stdG, stdR))
  }

  def apply(mean: Float, std: Float): ImageChannelNormalize = {
    new ImageChannelNormalize(Array(mean), Array(std))
  }
}

/**
 * Normalizes the norm or value range Per image, similar to opencv::normalize
 * https://docs.opencv.org/ref/master/d2/de8/group__core__array.html
 * #ga87eef7ee3970f86906d69a92cbf064bd
 * ImageNormalize normalizes scale and shift the input features.
 * Various normalize methods are supported. Eg. NORM_INF, NORM_L1,
 * NORM_L2 or NORM_MINMAX.
 * Pleas notice it's a per image normalization.
 * @param min lower range boundary in case of the range normalization or
 *            norm value to normalize
 * @param max upper range boundary in case of the range normalization;
 *            it is not used for the norm normalization.
 * @param normType normalization type, see opencv:NormTypes.
 *           https://docs.opencv.org/ref/master/d2/de8/group__core__array.html
 *           #gad12cefbcb5291cf958a85b4b67b6149f
 *           Default Core.NORM_MINMAX
 */
class PerImageNormalize(min: Double, max: Double, normType: Int = Core.NORM_MINMAX)
  extends ImageProcessing {
  override def transformMat(feature: ImageFeature): Unit = {
    PerImageNormalize.transform(feature.opencvMat(), feature.opencvMat(), min, max, normType)
  }
}

object PerImageNormalize {
  def apply(min: Double, max: Double, normType: Int = Core.NORM_MINMAX): PerImageNormalize = {
    new PerImageNormalize(min, max, normType)
  }

  def transform(input: OpenCVMat, output: OpenCVMat, min: Double, max: Double, normType: Int)
  : Unit = {
    Core.normalize(input, output, min, max, normType)
  }
}
