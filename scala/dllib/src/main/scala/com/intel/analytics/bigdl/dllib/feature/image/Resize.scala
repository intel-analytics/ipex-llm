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
