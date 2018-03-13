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
import org.opencv.core.CvType

/**
 * Pixel level normalizer, data(i) = data(i) - mean(i)
 *
 * @param means pixel level mean, following H * W * C order
 */
class PixelNormalizer(means: Array[Float]) extends FeatureTransformer {

  private var data: Array[Float] = _

  override def transformMat(feature: ImageFeature): Unit = {
    val openCVMat = feature.opencvMat()
    if (openCVMat.`type`() != CvType.CV_32FC3) {
      openCVMat.convertTo(openCVMat, CvType.CV_32FC3)
    }

    if (data == null) {
      data = new Array[Float](means.length)
    }
    require(data.length == openCVMat.height() * openCVMat.width() * openCVMat.channels(),
      s"the means (${means.length}) provided must have the same length as image" +
        s" ${openCVMat.height() * openCVMat.width() * openCVMat.channels()}")
    openCVMat.get(0, 0, data)

    require(means.length == data.length, s"Image size expected :" +
      s"${means.length}, actual : ${data.length}")

    var i = 0
    while (i < data.length) {
      data(i + 2) = data(i + 2) - means(i + 2)
      data(i + 1) = data(i + 1) - means(i + 1)
      data(i + 0) = data(i + 0) - means(i + 0)
      i += 3
    }

    openCVMat.put(0, 0, data)
  }

}

object PixelNormalizer {
  def apply(means: Array[Float]): PixelNormalizer = new PixelNormalizer(means)
}
