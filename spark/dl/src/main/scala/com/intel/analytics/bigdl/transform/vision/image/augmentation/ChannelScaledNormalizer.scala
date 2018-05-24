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

import com.intel.analytics.bigdl.dataset.image.LabeledBGRImage
import com.intel.analytics.bigdl.dataset.{LocalDataSet, Transformer}
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.apache.log4j.Logger

import scala.collection.Iterator

object ChannelScaledNormalizer {

  def apply(meanR: Int, meanG: Int, meanB: Int, scale: Double): ChannelScaledNormalizer = {
    new ChannelScaledNormalizer(meanR, meanG, meanB, scale)
  }
}

/**
 * Channel normalization with scale factor
 * @param meanR mean value for channel R
 * @param meanG mean value for channel G
 * @param meanB mean value for channel B
 * @param scale scale value applied for all channels
 */

class ChannelScaledNormalizer(meanR: Int, meanG: Int, meanB: Int, scale: Double)
  extends FeatureTransformer {

  override protected def transformMat(feature: ImageFeature): Unit = {
    val mat = feature.opencvMat()
    val toFloats = OpenCVMat.toFloatPixels(mat)
    val content = toFloats._1
    require(content.length % 3 == 0, "Content should be multiple of 3 channels")
    var i = 0
    val frameLength = content.length / 3
    val height = toFloats._2
    val width = toFloats._3
    val bufferContent = new Array[Float](width * height * 3)

    val channels = 3
    val mean = Array(meanR, meanG, meanB)
    var c = 0
    while (c < channels) {
      i = 0
      while (i < frameLength) {
        val data_index = c * frameLength + i
        bufferContent(data_index) = ((content(data_index) - mean(c)) * scale).toFloat
        i += 1
      }
      c += 1
    }
    if (mat != null) {
      mat.release()
    }
    val newMat = OpenCVMat.fromFloats(bufferContent, height, width)
    feature(ImageFeature.mat) = newMat
  }

}
