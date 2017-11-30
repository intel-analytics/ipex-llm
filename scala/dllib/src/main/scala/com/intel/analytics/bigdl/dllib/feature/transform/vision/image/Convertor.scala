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

package com.intel.analytics.bigdl.transform.vision.image

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.apache.log4j.Logger

/**
 * Transform byte array(original image file in byte) to OpenCVMat
 */
class BytesToMat()
  extends FeatureTransformer {

  override def transform(feature: ImageFeature): ImageFeature = {
    BytesToMat.transform(feature)
  }
}

object BytesToMat {
  val logger = Logger.getLogger(getClass)
  def apply(): BytesToMat = new BytesToMat()

  def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    val bytes = feature[Array[Byte]](ImageFeature.bytes)
    var mat: OpenCVMat = null
    try {
      require(null != bytes && bytes.length > 0, "image file bytes should not be empty")
      mat = OpenCVMat.fromImageBytes(bytes)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
    } catch {
      case e: Exception =>
        val uri = feature.uri()
        logger.warn(s"convert byte to mat fail for $uri")
        feature(ImageFeature.originalSize) = (-1, -1, -1)
        feature.isValid = false
    }
    feature
  }
}
