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
import org.opencv.imgproc.Imgproc

class ImageResize(
    resizeH: Int,
    resizeW: Int,
    resizeMode: Int = Imgproc.INTER_LINEAR,
    useScaleFactor: Boolean = true) extends ImageProcessing {

  private val internalResize = augmentation.Resize(resizeH, resizeW, resizeMode, useScaleFactor)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalResize.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalResize.transformMat(feature)
  }
}

object ImageResize {

  def apply(resizeH: Int, resizeW: Int,
            resizeMode: Int = Imgproc.INTER_LINEAR, useScaleFactor: Boolean = true): ImageResize =
    new ImageResize(resizeH, resizeW, resizeMode, useScaleFactor)

}

class ImageAspectScale(minSize: Int,
                       scaleMultipleOf: Int = 1,
                       maxSize: Int = 1000,
                       resizeMode: Int = Imgproc.INTER_LINEAR,
                       useScaleFactor: Boolean = true,
                       minScale: Option[Float] = None)
  extends ImageProcessing {

  private val internalCrop = augmentation.AspectScale(minSize, scaleMultipleOf,
    maxSize, resizeMode, useScaleFactor, minScale)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageAspectScale {
  def apply(minSize: Int,
            scaleMultipleOf: Int = 1,
            maxSize: Int = 1000,
            mode: Int = Imgproc.INTER_LINEAR,
            useScaleFactor: Boolean = true,
            minScale: Option[Float] = None): ImageAspectScale =
    new ImageAspectScale(minSize, scaleMultipleOf, maxSize, mode, useScaleFactor, minScale)
}

class ImageRandomAspectScale(scales: Array[Int], scaleMultipleOf: Int = 1,
                        maxSize: Int = 1000) extends ImageProcessing {

  private val internalCrop = augmentation.RandomAspectScale(scales, scaleMultipleOf, maxSize)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalCrop.transformMat(feature)
  }
}

object ImageRandomAspectScale {
  def apply(scales: Array[Int], scaleMultipleOf: Int = 1,
            maxSize: Int = 1000): ImageRandomAspectScale =
    new ImageRandomAspectScale(scales, scaleMultipleOf, maxSize)
}
