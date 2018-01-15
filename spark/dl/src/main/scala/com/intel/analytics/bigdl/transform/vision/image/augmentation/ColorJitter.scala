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

import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import scala.util.Random

/**
 * Random adjust brightness, contrast, hue, saturation
 *
 * @param brightnessProb probability to adjust brightness
 * @param brightnessDelta brightness parameter
 * @param contrastProb probability to adjust contrast
 * @param contrastLower contrast lower parameter
 * @param contrastUpper contrast upper parameter
 * @param hueProb probability to adjust hue
 * @param hueDelta hue parameter
 * @param saturationProb probability to adjust saturation
 * @param saturationLower saturation lower parameter
 * @param saturationUpper saturation upper parameter
 * @param randomChannelOrderProb random order for different operation
 * @param shuffle shuffle the transformers
 */
class ColorJitter(
  brightnessProb: Double, brightnessDelta: Double,
  contrastProb: Double, contrastLower: Double, contrastUpper: Double,
  hueProb: Double, hueDelta: Double,
  saturationProb: Double, saturationLower: Double, saturationUpper: Double,
  randomChannelOrderProb: Double, shuffle: Boolean = false) extends FeatureTransformer {

  private val brightness = RandomTransformer(
    Brightness(-brightnessDelta, brightnessDelta), brightnessProb)
  private val contrast = RandomTransformer(
    Contrast(contrastLower, contrastUpper), contrastProb)
  private val saturation = RandomTransformer(
    Saturation(saturationLower, saturationUpper), saturationProb)
  private val hue = RandomTransformer(Hue(-hueDelta, hueDelta), hueProb)
  private val channelOrder = RandomTransformer(ChannelOrder(), randomChannelOrderProb)

  private val order1 = brightness -> contrast -> saturation -> hue -> channelOrder
  private val order2 = brightness -> saturation -> hue -> contrast -> channelOrder

  private val transformers = Array(brightness, contrast, saturation, hue, channelOrder)

  override def transformMat(feature: ImageFeature): Unit = {
    if (!shuffle) {
      val prob = RNG.uniform(0, 1)
      if (prob > 0.5) {
        order1.transform(feature)
      } else {
        order2.transform(feature)
      }
    } else {
      val order = Random.shuffle(List.range(0, transformers.length))
      var i = 0
      while (i < order.length) {
        transformers(order(i)).transform(feature)
        i += 1
      }
    }
  }
}

object ColorJitter {
  def apply(
    brightnessProb: Double = 0.5, brightnessDelta: Double = 32,
    contrastProb: Double = 0.5, contrastLower: Double = 0.5, contrastUpper: Double = 1.5,
    hueProb: Double = 0.5, hueDelta: Double = 18,
    saturationProb: Double = 0.5, saturationLower: Double = 0.5, saturationUpper: Double = 1.5,
    randomOrderProb: Double = 0, shuffle: Boolean = false
  ): ColorJitter =
    new ColorJitter(brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb, shuffle)
}
