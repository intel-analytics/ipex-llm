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

package com.intel.analytics.bigdl.transform.vision.image.augmentation

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.RandomGenerator._

/**
 * adjust the image brightness
 *
 * @param deltaLow brightness parameter: low bound
 * @param deltaHigh brightness parameter: high bound
 */
class Brightness(deltaLow: Double, deltaHigh: Double)
  extends FeatureTransformer {
  require(deltaLow <= deltaHigh)
  override def transformMat(feature: ImageFeature): Unit = {
    Brightness.transform(feature.opencvMat(), feature.opencvMat(), RNG.uniform(deltaLow, deltaHigh))
  }
}

object Brightness {
  def apply(deltaLow: Double, deltaHigh: Double): Brightness
  = new Brightness(deltaLow, deltaHigh)

  /**
   * if delta > 0, increase the brightness
   * if delta < 0, decrease the brightness
   * @param input input mat
   * @param output output mat
   * @param delta brightness parameter
   * @return output mat
   */
  def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
    if (delta != 0) {
      input.convertTo(output, -1, 1, delta)
    } else {
      if (input != output) input.copyTo(output)
    }
    output
  }
}
