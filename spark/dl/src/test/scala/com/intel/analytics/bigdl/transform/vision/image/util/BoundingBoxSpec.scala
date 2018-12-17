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

package com.intel.analytics.bigdl.transform.vision.image.util

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class BoundingBoxSpec extends FlatSpec with Matchers {
  "scaleBBox" should "work properly" in {
    val bbox = BoundingBox(1f, 4f, 5f, 6f, false)
    val scaled = new BoundingBox()
    bbox.scaleBox(1.0f / 4, 1.0f / 2, scaled)

    scaled should be(BoundingBox(0.5f, 1, 2.5f, 1.5f))
  }

  "meetEmitCenterConstraint" should "work properly" in {
    val bbox = BoundingBox(0, 0, 5, 3, false)
    val bbox2 = BoundingBox(1, 0, 7, 4, false)
    bbox.meetEmitCenterConstraint(bbox2) should be(true)
  }

  "meetEmitCenterConstraint false" should "work properly" in {
    val bbox = BoundingBox(0, 0, 5, 3, false)
    val bbox2 = BoundingBox(4, 0, 7, 4, false)
    bbox.meetEmitCenterConstraint(bbox2) should be(false)
  }

  "jaccardOverlap partial overlap" should "work properly" in {
    val bbox1 = BoundingBox(0.2f, 0.3f, 0.3f, 0.5f)
    val bbox2 = BoundingBox(0.1f, 0.1f, 0.3f, 0.4f)
    val overlap = bbox1.jaccardOverlap(bbox2)
    assert(Math.abs(overlap - 1.0 / 7) < 1e-6)
  }

  "jaccardOverlap fully contain" should "work properly" in {
    val bbox1 = BoundingBox(0.2f, 0.3f, 0.3f, 0.5f)
    val bbox2 = BoundingBox(0.1f, 0.1f, 0.4f, 0.6f)
    val overlap = bbox1.jaccardOverlap(bbox2)
    assert(Math.abs(overlap - 2.0 / 15) < 1e-6)
  }

  "jaccardOverlap outside" should "work properly" in {
    val bbox1 = BoundingBox(0.2f, 0.3f, 0.3f, 0.5f)
    val bbox2 = BoundingBox(0f, 0f, 0.1f, 0.1f)
    val overlap = bbox1.jaccardOverlap(bbox2)
    assert(Math.abs(overlap - 0) < 1e-6)
  }

  "projectBbox" should "work properly" in {
    val box1 = BoundingBox(0.222159f, 0.427017f, 0.606492f, 0.679355f)
    val box2 = BoundingBox(0.418f, 0.396396f, 0.55f, 0.666667f)
    val projBox = new BoundingBox()
    val state = box1.projectBbox(box2, projBox)
    state should be(true)
    assert(Math.abs(projBox.x1 - 0.509561f) < 1e-5)
    assert(Math.abs(projBox.y1 - 0f) < 1e-5)
    assert(Math.abs(projBox.x2 - 0.853014f) < 1e-5)
    assert(Math.abs(projBox.y2 - 0.949717f) < 1e-5)
  }

  "meetEmitCenterConstraint true" should "work properly" in {
    val box1 = BoundingBox(0.222159f, 0.427017f, 0.606492f, 0.679355f)
    val box2 = BoundingBox(0.418f, 0.396396f, 0.55f, 0.666667f)

    val state = box1.meetEmitCenterConstraint(box2)

    state should be(true)
  }

  "meetEmitCenterConstraint normalized false" should "work properly" in {
    val box1 = BoundingBox(0.0268208f, 0.388175f, 0.394421f, 0.916685f)
    val box2 = BoundingBox(0.418f, 0.396396f, 0.55f, 0.666667f)

    val state = box1.meetEmitCenterConstraint(box2)

    state should be(false)
  }

  "getLocPredictions shared" should "work properly" in {
    val num = 2
    val numPredsPerClass = 2
    val numLocClasses = 1
    val shareLoc = true
    val dim = numPredsPerClass * numLocClasses * 4
    val loc = Tensor[Float](num, dim, 1, 1)

    val locData = loc.storage().array()
    (0 until num).foreach(i => {
      (0 until numPredsPerClass).foreach(j => {
        val idx = i * dim + j * 4
        locData(idx) = i * numPredsPerClass * 0.1f + j * 0.1f
        locData(idx + 1) = i * numPredsPerClass * 0.1f + j * 0.1f
        locData(idx + 2) = i * numPredsPerClass * 0.1f + j * 0.1f + 0.2f
        locData(idx + 3) = i * numPredsPerClass * 0.1f + j * 0.1f + 0.2f
      })
    })

    val out = BboxUtil.getLocPredictions(loc, numPredsPerClass, numLocClasses, shareLoc)

    assert(out.length == num)

    (0 until num).foreach(i => {
      assert(out(i).length == 1)
      val bboxes = out(i)(0)
      assert(bboxes.size(1) == numPredsPerClass)
      val startValue = i * numPredsPerClass * 0.1f
      var j = 0
      while (j < numPredsPerClass) {
        expectNear(bboxes(j + 1).valueAt(1), startValue + j * 0.1, 1e-6)
        expectNear(bboxes(j + 1).valueAt(2), startValue + j * 0.1, 1e-6)
        expectNear(bboxes(j + 1).valueAt(3), startValue + j * 0.1 + 0.2, 1e-6)
        expectNear(bboxes(j + 1).valueAt(4), startValue + j * 0.1 + 0.2, 1e-6)
        j += 1
      }
    })
  }

  def expectNear(v1: Float, v2: Double, eps: Double): Unit = {
    assert(Math.abs(v1 - v2) < eps)
  }

  "decodeBoxes" should "work properly" in {
    val priorBoxes = Tensor[Float](4, 4)
    val priorVariances = Tensor[Float](4, 4)
    val bboxes = Tensor[Float](4, 4)
    var i = 1
    while (i < 5) {
      priorBoxes.setValue(i, 1, 0.1f * i)
      priorBoxes.setValue(i, 2, 0.1f * i)
      priorBoxes.setValue(i, 3, 0.1f * i + 0.2f)
      priorBoxes.setValue(i, 4, 0.1f * i + 0.2f)

      priorVariances.setValue(i, 1, 0.1f)
      priorVariances.setValue(i, 2, 0.1f)
      priorVariances.setValue(i, 3, 0.2f)
      priorVariances.setValue(i, 4, 0.2f)

      bboxes.setValue(i, 1, 0f)
      bboxes.setValue(i, 2, 0.75f)
      bboxes.setValue(i, 3, Math.log(2).toFloat)
      bboxes.setValue(i, 4, Math.log(3f / 2).toFloat)
      i += 1
    }

    val decodedBboxes = BboxUtil.decodeBoxes(priorBoxes, priorVariances, false, bboxes, true)

    assert(decodedBboxes.size(1) == 4)

    i = 1
    while (i < 5) {
      expectNear(decodedBboxes.valueAt(i, 1), 0 + (i - 1) * 0.1, 1e-5)
      expectNear(decodedBboxes.valueAt(i, 2), 0.2 + (i - 1) * 0.1, 1e-5)
      expectNear(decodedBboxes.valueAt(i, 3), 0.4 + (i - 1) * 0.1, 1e-5)
      expectNear(decodedBboxes.valueAt(i, 4), 0.5 + (i - 1) * 0.1, 1e-5)
      i += 1
    }

  }


  "getPriorVariance" should "work properly" in {
    val num_channels = 2
    val num_priors = 2
    val dim = num_priors * 4
    val prior = Tensor[Float](1, num_channels, dim, 1)
    val prior_data = prior.storage().array()
    for (i <- 0 until num_priors) {
      prior_data(i * 4) = i * 0.1f
      prior_data(i * 4 + 1) = i * 0.1f
      prior_data(i * 4 + 2) = i * 0.1f + 0.2f
      prior_data(i * 4 + 3) = i * 0.1f + 0.1f
      for (j <- 0 until 4) {
        prior_data(dim + i * 4 + j) = 0.1f
      }
    }

    val (boxes, variances) = BboxUtil.getPriorBboxes(prior, num_priors)
    assert(boxes.size(1) == num_priors)
    assert(variances.size(1) == num_priors)
    for (i <- 0 until num_priors) {
      expectNear(boxes.valueAt(i + 1, 1), i * 0.1, 1e-5)
      expectNear(boxes.valueAt(i + 1, 2), i * 0.1, 1e-5)
      expectNear(boxes.valueAt(i + 1, 3), i * 0.1 + 0.2, 1e-5)
      expectNear(boxes.valueAt(i + 1, 4), i * 0.1 + 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 1), 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 2), 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 3), 0.1, 1e-5)
      expectNear(variances.valueAt(i + 1, 4), 0.1, 1e-5)
    }
  }

  "getGroundTruths" should "work properly" in {
    val input = Tensor(Storage(Array(
      0.0f, 1.0f, 0.14285715f, 0.1904762f, 0.23809524f, 0.2857143f, 0.33333334f,
      0.0f, 1.0f, 0.47619048f, 0.52380955f, 0.5714286f, 0.61904764f, 0.6666667f,
      1.0f, 3.0f, 0.8095238f, 0.85714287f, 0.9047619f, 0.95238096f, 1.0f
    ))).resize(3, 7)

    val gt0 = Tensor(Storage(Array(
      0.0f, 1.0f, 0.14285715f, 0.1904762f, 0.23809524f, 0.2857143f, 0.33333334f,
      0.0f, 1.0f, 0.47619048f, 0.52380955f, 0.5714286f, 0.61904764f, 0.6666667f
    ))).resize(2, 7)

    val gt1 = Tensor(Storage(Array(
      1.0f, 3.0f, 0.8095238f, 0.85714287f, 0.9047619f, 0.95238096f, 1.0f
    ))).resize(1, 7)

    val gts = BboxUtil.getGroundTruths(input)

    gts(0) should be(gt0)
    gts(1) should be(gt1)

    val gts2 = BboxUtil.getGroundTruths(gt1)

    gts2(0) should be(gt1)

    val label = Tensor(Storage(Array(
      3.0, 8.0, 0.0, 0.241746, 0.322738, 0.447184, 0.478388,
      3.0, 8.0, 0.0, 0.318659, 0.336546, 0.661729, 0.675461,
      3.0, 8.0, 0.0, 0.56154, 0.300144, 0.699173, 0.708098,
      3.0, 8.0, 0.0, 0.220494, 0.327759, 0.327767, 0.396797,
      3.0, 8.0, 0.0, 0.194182, 0.317717, 0.279191, 0.389266,
      4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      6.0, 10.0, 0.0, 0.67894, 0.471823, 0.929308, 0.632044,
      6.0, 10.0, 0.0, 0.381443, 0.572376, 0.892489, 0.691713,
      7.0, 9.0, 0.0, 0.0, 0.0620616, 0.667269, 1.0
    ).map(_.toFloat))).resize(10, 7)


    val labelgt = BboxUtil.getGroundTruths(label)

    labelgt.size should be(3)
    labelgt(0).size(1) should be(5)
    labelgt(0).valueAt(1, 1) should be(3)
    labelgt(3).size(1) should be(2)
    labelgt(3).valueAt(1, 1) should be(6)
    labelgt(4).size(1) should be(1)
    labelgt(4).valueAt(1, 1) should be(7)
  }
}
