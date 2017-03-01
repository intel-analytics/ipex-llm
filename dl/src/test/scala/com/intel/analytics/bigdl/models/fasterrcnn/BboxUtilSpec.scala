/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.fasterrcnn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class BboxUtilSpec extends FlatSpec with Matchers {
  "bboxTransformInv" should "work properly" in {

    val boxes = Tensor(Storage(Array(0.54340494, 0.2783694, 0.4245176, 0.84477615,
      0.0047188564, 0.12156912, 0.67074907, 0.82585275,
      0.13670659, 0.5750933, 0.89132196, 0.20920213,
      0.18532822, 0.10837689, 0.21969749, 0.9786238,
      0.8116832, 0.17194101, 0.81622475, 0.27407375).map(x => x.toFloat))).resize(5, 4)



    val deltas = Tensor(Storage(Array(
      0.4317042, 0.9400298, 0.81764936, 0.33611196,
      0.7956625, 0.015254972, 0.5988434, 0.6038045,
      0.98092085, 0.05994199, 0.89054596, 0.5769015,
      0.21002658, 0.5446849, 0.76911515, 0.25069523,
      0.35950786, 0.59885895, 0.3547956, 0.3401902)
      .map(x => x.toFloat)))
    deltas.resize(5, 4)

    val expectedResults = Tensor(Storage(Array(
      0.36640674, 1.437952, 2.3622758, 3.6301315,
      0.64723384, -0.55891246, 3.6794298, 2.558332,
      0.5976285, 0.36563802, 4.872678, 1.494677,
      -0.19625212, 0.8606382, 2.0357678, 3.263753,
      0.958912, 0.6086628, 2.391277, 2.157396)
      .map(x => x.toFloat)))
    expectedResults.resize(5, 4)

    val res = BboxUtil.bboxTransformInv(boxes, deltas)
    res should be(expectedResults)
  }

  "clipBoxes" should "work properly" in {
    val boxes = Tensor(Storage(Array(
      43.170418, 94.00298, 81.76494, 33.611195,
      79.56625, 1.5254971, 59.88434, 60.380455,
      98.09209, 5.994199, 89.054596, 57.69015,
      21.002657, 35.468487, 76.911514, 55.069523,
      35.950783, 34.885895, 35.47956, 70.01902)
      .map(x => x.toFloat))).resize(5, 4)

    val scores = Tensor(Storage(Array(0.999516, 0.9487129, 0.9859998, 0.9985473, 0.9780578)
      .map(x => x.toFloat)))

    val expectedResults = Tensor(Storage(Array(
      43.170418, 59.0, 49.0, 33.611195,
      49.0, 1.5254971, 49.0, 59.0,
      49.0, 5.994199, 49.0, 57.69015,
      21.002657, 35.468487, 49.0, 55.069523,
      35.950783, 34.885895, 35.47956, 59.0)
      .map(x => x.toFloat))).resize(5, 4)

    val expectedScores = Tensor(Storage(Array(0, 0, 0, 0.9985473, 0).map(x => x.toFloat)))

    BboxUtil.clipBoxes(boxes, 60, 50, 4, 5, scores)
    scores should be(expectedScores)
    boxes should be(expectedResults)
  }
}
