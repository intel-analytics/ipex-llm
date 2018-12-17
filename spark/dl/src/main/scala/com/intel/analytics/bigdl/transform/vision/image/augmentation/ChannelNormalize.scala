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
 *
 * @param means mean value in each channel
 * @param stds std value in each channel
 */
class ChannelNormalize(means: Array[Float], stds: Array[Float])
  extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    ChannelNormalize.transform(feature.opencvMat(), feature.opencvMat(),
      means, stds)
  }
}

object ChannelNormalize {
  /**
   * image channel normalize
   *
   * @param meanR mean value in R channel
   * @param meanG mean value in G channel
   * @param meanB mean value in B channel
   * @param stdR std value in R channel
   * @param stdG std value in G channel
   * @param stdB std value in B channel
   */
  def apply(meanR: Float, meanG: Float, meanB: Float,
    stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): ChannelNormalize = {
    new ChannelNormalize(Array(meanB, meanG, meanR), Array(stdR, stdG, stdB))
  }

  def apply(mean: Float, std: Float): ChannelNormalize = {
    new ChannelNormalize(Array(mean), Array(std))
  }

  def transform(input: OpenCVMat, output: OpenCVMat, means: Array[Float], stds: Array[Float])
  : Unit = {
    val channel = input.channels()
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

    (0 until channel).foreach(i => {
      if (null != means) {
        Core.subtract(inputChannels.get(i), new Scalar(means(i)), outputChannels.get(i))
      }
      if (stds != null) {
        if (stds(i) != 1) {
          Core.divide(outputChannels.get(i), new Scalar(stds(i)), outputChannels.get(i))
        }
      }
    })
    Core.merge(outputChannels, output)

    (0 until inputChannels.size()).foreach(inputChannels.get(_).release())
    if (input != output) {
      (0 until outputChannels.size()).foreach(outputChannels.get(_).release())
    }
  }
}
