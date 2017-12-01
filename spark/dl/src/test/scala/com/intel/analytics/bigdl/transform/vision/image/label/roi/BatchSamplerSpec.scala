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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

class BatchSamplerSpec extends FlatSpec with Matchers {
  "batch sampler with no change" should "work properly" in {
    val sampler = new BatchSampler(maxTrials = 1)
    val unitBox = BoundingBox(0, 0, 1, 1)
    val boxes = Tensor(Storage(Array(0.582296, 0.334719, 0.673582, 0.52183,
      0.596127, 0.282744, 0.670816, 0.449064,
      0.936376, 0.627859, 0.961272, 0.733888,
      0.896266, 0.640333, 0.923928, 0.740125).map(x => x.toFloat))).resize(4, 4)
    val classes = Tensor[Float](4).randn()
    val target = RoiLabel(classes, boxes)
    val sampledBoxes = new ArrayBuffer[BoundingBox]()
    sampler.sample(unitBox, target, sampledBoxes)

    sampledBoxes.length should be(1)
    sampledBoxes(0) should be(unitBox)
  }

  "satisfySampleConstraint with minOverlap 0.1" should "work properly" in {
    val boxes = Tensor(Storage(Array(0.418, 0.396396, 0.55, 0.666667,
      0.438, 0.321321, 0.546, 0.561562,
      0.93, 0.81982, 0.966, 0.972973,
      0.872, 0.837838, 0.912, 0.981982).map(x => x.toFloat))).resize(4, 4)
    val classes = Tensor[Float](4).randn()
    val target = RoiLabel(classes, boxes)

    val sampledBox = BoundingBox(0.114741f, 0.248062f, 0.633665f, 0.763736f)
    val sampler = new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.1))

    sampler.satisfySampleConstraint(sampledBox, target) should be(true)
  }

  "satisfySampleConstraint with minOverlap 0.3" should "work properly" in {
    val boxes = Tensor(Storage(Array(0.418, 0.396396, 0.55, 0.666667,
      0.438, 0.321321, 0.546, 0.561562,
      0.93, 0.81982, 0.966, 0.972973,
      0.872, 0.837838, 0.912, 0.981982).map(x => x.toFloat))).resize(4, 4)
    val classes = Tensor[Float](4).randn()
    val target = RoiLabel(classes, boxes)

    val sampledBox = BoundingBox(0.266885f, 0.416113f, 0.678256f, 0.67208f)
    val sampler = new BatchSampler(minScale = 0.3, minAspectRatio = 0.5, maxAspectRatio = 2,
      minOverlap = Some(0.3))

    sampler.satisfySampleConstraint(sampledBox, target) should be(true)
  }

  "batch samplers" should "work properly" in {
    val boxes = Tensor(Storage(Array(0.418, 0.396396, 0.55, 0.666667,
      0.438, 0.321321, 0.546, 0.561562,
      0.93, 0.81982, 0.966, 0.972973,
      0.872, 0.837838, 0.912, 0.981982).map(x => x.toFloat))).resize(4, 4)
    val classes = Tensor[Float](4).randn()
    val target = RoiLabel(classes, boxes)
    val sampledBoxes = new ArrayBuffer[BoundingBox]()
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
    BatchSampler.generateBatchSamples(target, batchSamplers, sampledBoxes)

    sampledBoxes.foreach(box => {
      println(box)
    })
  }
}
