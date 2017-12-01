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

package com.intel.analytics.bigdl.transform.vision.image.label.roi

import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Crop
import com.intel.analytics.bigdl.transform.vision.image.util.{BoundingBox}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.opencv.core.Mat

import scala.collection.mutable.ArrayBuffer

/**
 * Random sample a bounding box given some constraints and crop the image
 * This is used in SSD training augmentation
 */
class RandomSampler extends Crop {
  // random cropping samplers
  val batchSamplers = Array(
    new BatchSampler(maxTrials = 1),
    new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.1)),
    new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.3)),
    new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.5)),
    new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.7)),
    new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.9)),
    new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      maxOverlap = Some(1.0)))

  def generateRoi(feature: ImageFeature): BoundingBox = {
    val roiLabel = feature(ImageFeature.label).asInstanceOf[RoiLabel]
    val boxesBuffer = new ArrayBuffer[BoundingBox]()
    BatchSampler.generateBatchSamples(roiLabel,
      batchSamplers, boxesBuffer)

    // randomly pick up one as input data
    if (boxesBuffer.nonEmpty) {
      // Randomly pick a sampled bbox and crop the expand_datum.
      val index = (RNG.uniform(0, 1) * boxesBuffer.length).toInt
      boxesBuffer(index)
    } else {
      BoundingBox(0, 0, 1, 1)
    }
  }
}

object RandomSampler {
  def apply(): FeatureTransformer = {
    new RandomSampler() -> RoiProject()
  }
}
