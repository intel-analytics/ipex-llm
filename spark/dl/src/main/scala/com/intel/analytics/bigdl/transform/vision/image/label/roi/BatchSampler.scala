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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.util.{BboxUtil, BoundingBox}
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.collection.mutable.ArrayBuffer


/**
 * sample box from given parameters, and regard it as positive if it satisfies overlap
 * constraints
 *
 * @param maxSample maximum random crop samples to be generated
 * @param maxTrials maximum trials, if exceed this number, give up anyway
 * @param minScale min scale
 * @param maxScale max scale
 * @param minAspectRatio min aspect ratio
 * @param maxAspectRatio max aspect ratio
 * @param minOverlap min overlap between sampled box and gt box
 * @param maxOverlap max overlap between sampled box and gt box
 */
class BatchSampler(maxSample: Int = 1, maxTrials: Int = 50,
  minScale: Double = 1, maxScale: Double = 1,
  minAspectRatio: Double = 1, maxAspectRatio: Double = 1,
  minOverlap: Option[Double] = None,
  maxOverlap: Option[Double] = None) extends Serializable {

  require(minScale <= maxScale, "minScale must <= maxScale")
  require(minScale > 0 && minScale <= 1, "minScale must in (0, 1]")
  require(maxScale > 0 && maxScale <= 1, "maxScale must in (0, 1]")
  require(minAspectRatio > 0 && minAspectRatio <= 1, "minAspectRatio must in (0, 1]")
  require(maxAspectRatio >= 1, "minAspectRatio must >= 1")
  if (minOverlap.isDefined) {
    require(minOverlap.get >= 0 && minOverlap.get <= 1, "minOverlap must in [0, 1]")
  }

  def satisfySampleConstraint(sampledBox: BoundingBox, target: RoiLabel): Boolean = {
    // By default, the sampled_bbox is "positive" if no constraints are defined.
    if (minOverlap.isEmpty && maxOverlap.isEmpty) return true
    var i = 1
    while (i <= target.size()) {
      val overlap = jaccardOverlap(sampledBox, target.bboxes, i)
      if (minOverlap.isEmpty || overlap >= minOverlap.get) {
        if (maxOverlap.isEmpty || overlap <= maxOverlap.get) {
          return true
        }
      }
      i += 1
    }
    false
  }

  def sample(sourceBox: BoundingBox, target: RoiLabel, sampledBoxes: ArrayBuffer[BoundingBox])
  : Unit = {
    var found = 0
    var trial = 0
    while (trial < maxTrials) {
      if (found >= maxSample) {
        return
      }
      // Generate sampled_bbox in the normalized space [0, 1].
      val sampledBox = sampleBox()
      // Transform the sampled_bbox w.r.t. source_bbox.
      sourceBox.locateBBox(sampledBox, sampledBox)
      // Determine if the sampled bbox is positive or negative by the constraint.
      if (satisfySampleConstraint(sampledBox, target)) {
        found += 1
        sampledBoxes.append(sampledBox)
      }
      trial += 1
    }
  }

  private def sampleBox(): BoundingBox = {
    val scale = RNG.uniform(minScale, maxScale)
    var ratio = RNG.uniform(minAspectRatio, maxAspectRatio)
    ratio = Math.max(ratio, scale * scale)
    ratio = Math.min(ratio, 1 / scale / scale)
    val width = scale * Math.sqrt(ratio)
    val height = scale / Math.sqrt(ratio)
    val x1 = RNG.uniform(0, 1 - width).toFloat
    val y1 = RNG.uniform(0, 1 - height).toFloat
    val x2 = x1 + width.toFloat
    val y2 = y1 + height.toFloat
    BoundingBox(x1, y1, x2, y2)
  }

  private def jaccardOverlap(bbox: BoundingBox, gtBoxes: Tensor[Float], i: Int): Float = {
    val gtBox = BoundingBox(gtBoxes.valueAt(i, 1),
      gtBoxes.valueAt(i, 2),
      gtBoxes.valueAt(i, 3),
      gtBoxes.valueAt(i, 4))

    bbox.jaccardOverlap(gtBox)
  }

}

object BatchSampler {

  /**
   * generate batch samples
   * @param label normalized
   * @param batchSamplers
   * @param sampledBoxes
   */
  def generateBatchSamples(label: RoiLabel, batchSamplers: Array[BatchSampler],
    sampledBoxes: ArrayBuffer[BoundingBox]): Unit = {
    sampledBoxes.clear()
    var i = 0
    val unitBox = BoundingBox(0, 0, 1, 1)
    while (i < batchSamplers.length) {
      batchSamplers(i).sample(unitBox, label, sampledBoxes)
      i += 1
    }
  }
}
