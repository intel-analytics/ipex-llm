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
import org.opencv.core.Core

/**
 * Flip the image horizontally
 */
class HFlip() extends FeatureTransformer {

  override def transformMat(feature: ImageFeature): Unit = {
    HFlip.transform(feature.opencvMat(), feature.opencvMat())
  }
}

object HFlip {
  def apply(): HFlip = new HFlip()

  def transform(input: OpenCVMat, output: OpenCVMat): OpenCVMat = {
    Core.flip(input, output, 1)
    output
  }
}
