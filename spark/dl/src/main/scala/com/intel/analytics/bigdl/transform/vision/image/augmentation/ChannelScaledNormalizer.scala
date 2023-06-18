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
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import java.util
import org.apache.log4j.Logger
import org.opencv.core.{Core, CvType, Mat, Scalar}
import scala.collection.Iterator

object ChannelScaledNormalizer {

  def apply(meanR: Int, meanG: Int, meanB: Int, scale: Double): ChannelScaledNormalizer = {
    new ChannelScaledNormalizer(meanR, meanG, meanB, scale)
  }
  def transform(input: OpenCVMat, output: OpenCVMat,
    meanR: Int, meanG: Int, meanB: Int, scale: Double): Unit = {
    val channel = input.channels()
    require(channel == 3, s"Number of channel $channel != 3")
    if (input.`type`() != CvType.CV_32FC(channel)) {
      input.convertTo(input, CvType.CV_32FC(channel))
    }
    val inputChannels = new util.ArrayList[Mat]()
    Core.split(input, inputChannels)
    val outputChannels = if (output != input) {
      output.create(input.rows(), input.cols(), input.`type`())
      val channels = new util.ArrayList[Mat]()
      Core.split(output, channels)
      channels
    } else inputChannels

    val means = Array(meanB, meanG, meanR)
    (0 until channel).foreach(i => {
      Core.subtract(inputChannels.get(i), new Scalar(means(i)), outputChannels.get(i))
      Core.multiply(outputChannels.get(i), new Scalar(scale), outputChannels.get(i))
    })
    Core.merge(outputChannels, output)

    (0 until inputChannels.size()).foreach(inputChannels.get(_).release())
    if (input != output) {
      (0 until outputChannels.size()).foreach(outputChannels.get(_).release())
    }
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
    ChannelScaledNormalizer.transform(feature.opencvMat(), feature.opencvMat(),
      meanR, meanG, meanB, scale)
  }

}
