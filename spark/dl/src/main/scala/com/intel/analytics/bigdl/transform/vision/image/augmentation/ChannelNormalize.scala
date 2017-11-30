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
import org.opencv.core.{Core, CvType, Mat, Scalar}

/**
 * image channel normalize
 * @param meanR mean value in R channel
 * @param meanG mean value in G channel
 * @param meanB mean value in B channel
 * @param stdR std value in R channel
 * @param stdG std value in G channel
 * @param stdB std value in B channel
 */
class ChannelNormalize(meanR: Float, meanG: Float, meanB: Float,
                       stdR: Float = 1, stdG: Float = 1, stdB: Float = 1)
  extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    ChannelNormalize.transform(feature.opencvMat(), feature.opencvMat(),
      meanR, meanG, meanB, stdR, stdG, stdB)
  }
}

object ChannelNormalize {
  def apply(meanR: Float, meanG: Float, meanB: Float,
            stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): ChannelNormalize = {
    new ChannelNormalize(meanR, meanG, meanB, stdR, stdG, stdB)
  }

  def transform(input: OpenCVMat, output: OpenCVMat,
                meanR: Float, meanG: Float, meanB: Float,
                stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): Unit = {
    if (input.`type`() != CvType.CV_32FC3) {
      input.convertTo(input, CvType.CV_32FC3)
    }
    val inputChannels = new util.ArrayList[Mat]()
    Core.split(input, inputChannels)
    require(inputChannels.size() == 3)
    val outputChannels = if (output != input) {
      output.create(input.rows(), input.cols(), input.`type`())
      val channels = new util.ArrayList[Mat]()
      Core.split(output, channels)
      channels
    } else inputChannels

    Core.subtract(inputChannels.get(0), new Scalar(meanB), outputChannels.get(0))
    Core.subtract(inputChannels.get(1), new Scalar(meanG), outputChannels.get(1))
    Core.subtract(inputChannels.get(2), new Scalar(meanR), outputChannels.get(2))
    if (stdB != 1) Core.divide(outputChannels.get(0), new Scalar(stdB), outputChannels.get(0))
    if (stdG != 1) Core.divide(outputChannels.get(1), new Scalar(stdG), outputChannels.get(1))
    if (stdR != 1) Core.divide(outputChannels.get(2), new Scalar(stdR), outputChannels.get(2))
    Core.merge(outputChannels, output)

    (0 until inputChannels.size()).foreach(inputChannels.get(_).release())
    if (input != output) {
      (0 until outputChannels.size()).foreach(outputChannels.get(_).release())
    }
  }
}
