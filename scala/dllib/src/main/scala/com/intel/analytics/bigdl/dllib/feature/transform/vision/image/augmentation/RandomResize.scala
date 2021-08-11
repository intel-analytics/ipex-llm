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

import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.opencv.imgproc.Imgproc

object RandomResize {
  def apply(minSize: Int, maxSize: Int): RandomResize = new RandomResize(minSize, maxSize)
}

/**
 * Random resize between minSize and maxSize and scale height and width to each other
 * @param minSize min size to resize to
 * @param maxSize max size to resize to
 */
class RandomResize(minSize: Int, maxSize : Int) extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    if (feature.isValid) {
      var height = feature.opencvMat.height
      var width = feature.opencvMat.width
      val shorterSize = RandomGenerator.RNG.uniform(1e-2, maxSize - minSize + 1).toInt + minSize
      if (height < width) {
        width = (width.toFloat / height * shorterSize).toInt
        height = shorterSize
      } else {
        height = (height.toFloat / width * shorterSize).toInt
        width = shorterSize
      }

      Resize.transform(feature.opencvMat(), feature.opencvMat(), width, height, Imgproc.INTER_CUBIC)
    }
  }
}
