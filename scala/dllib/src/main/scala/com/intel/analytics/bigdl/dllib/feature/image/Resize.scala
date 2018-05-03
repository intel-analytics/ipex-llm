package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.bigdl.transform.vision.image.augmentation
import org.opencv.imgproc.Imgproc

class Resize(
    resizeH: Int,
    resizeW: Int,
    resizeMode: Int = Imgproc.INTER_LINEAR,
    useScaleFactor: Boolean = true
  ) extends Preprocessing[ImageFeature, ImageFeature] {
  
  private val internalResize = augmentation.Resize(resizeH, resizeW)
  def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalResize.apply(prev)
  }
}

object Resize {

  def apply(resizeH: Int, resizeW: Int,
            resizeMode: Int = Imgproc.INTER_LINEAR, useScaleFactor: Boolean = true): Resize =
    new Resize(resizeH, resizeW, resizeMode, useScaleFactor)

}