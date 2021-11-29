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
package com.intel.analytics.bigdl.dllib.feature.image

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{FeatureTransformer, ImageFeature, augmentation}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.augmentation.Crop
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.label.roi.{BatchSampler, RandomSampler, RoiLabel, RoiProject}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.util.BoundingBox
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import org.opencv.core.Mat

import scala.collection.mutable.ArrayBuffer

/**
 * Random sample a bounding box given some constraints and crop the image
 * This is used in SSD training augmentation
 */
class ImageRandomSampler extends ImageProcessing {
  private val internalSampler = RandomSampler()
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalSampler.apply(prev)
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    internalSampler.transform(feature)
  }
}

object ImageRandomSampler {
  def apply(): ImageRandomSampler = {
    new ImageRandomSampler()
  }
}
