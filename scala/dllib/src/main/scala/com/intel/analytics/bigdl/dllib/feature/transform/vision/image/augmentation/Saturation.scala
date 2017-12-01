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

import java.util

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.opencv.core.{Core, Mat}
import org.opencv.imgproc.Imgproc

/**
 * Adjust image saturation
 */
class Saturation(deltaLow: Double, deltaHigh: Double)
  extends FeatureTransformer {

  require(deltaHigh >= deltaLow, "saturation upper must be >= lower.")
  require(deltaLow >= 0, "saturation lower must be non-negative.")
  override def transformMat(feature: ImageFeature): Unit = {
    Saturation.transform(feature.opencvMat(), feature.opencvMat(), RNG.uniform(deltaLow, deltaHigh))
  }
}

object Saturation {
  def apply(deltaLow: Double, deltaHigh: Double): Saturation = new Saturation(deltaLow, deltaHigh)

  def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
    if (Math.abs(delta - 1) != 1e-3) {
      // Convert to HSV colorspace
      Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2HSV)

      // Split the image to 3 channels.
      val channels = new util.ArrayList[Mat]()
      Core.split(output, channels)

      // Adjust the saturation.
      channels.get(1).convertTo(channels.get(1), -1, delta, 0)
      Core.merge(channels, output)
      (0 until channels.size()).foreach(channels.get(_).release())
      // Back to BGR colorspace.
      Imgproc.cvtColor(output, output, Imgproc.COLOR_HSV2BGR)
    } else {
      if (input != output) input.copyTo(output)
    }
    output
  }
}
