package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, augmentation}
import com.intel.analytics.zoo.feature.common.Preprocessing

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
