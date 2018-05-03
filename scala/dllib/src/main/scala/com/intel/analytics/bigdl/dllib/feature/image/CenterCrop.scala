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

