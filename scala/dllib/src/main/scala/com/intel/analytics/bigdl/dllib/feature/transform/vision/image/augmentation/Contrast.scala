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
 * Adjust the image contrast
 * @param deltaLow contrast parameter low bound
 * @param deltaHigh contrast parameter high bound
 */
class Contrast(deltaLow: Double, deltaHigh: Double)
  extends FeatureTransformer {

  require(deltaHigh >= deltaLow, "contrast upper must be >= lower.")
  require(deltaLow >= 0, "contrast lower must be non-negative.")
  override def transformMat(feature: ImageFeature): Unit = {
    Contrast.transform(feature.opencvMat(), feature.opencvMat(), RNG.uniform(deltaLow, deltaHigh))
  }
}

object Contrast {
  def apply(deltaLow: Double, deltaHigh: Double): Contrast = new Contrast(deltaLow, deltaHigh)

  def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
    if (Math.abs(delta - 1) > 1e-3) {
      input.convertTo(output, -1, delta, 0)
    } else {
      if (input != output) input.copyTo(output)
    }
    output
  }
}
