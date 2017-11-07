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

package com.intel.analytics.bigdl.nn

import breeze.numerics.abs
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class SmoothL1CriterionWithWeightsSpec extends FlatSpec with Matchers {
  val inputArr = Array(1.1064437627792358398, -0.84479117393493652344, 0.10066220909357070923,
    0.41965752840042114258, 1.2952491044998168945, 0.19362208247184753418,
    0.19648919999599456787, -0.030720530077815055847)
  val targetArr = Array(0.97709864377975463867, 1.5001486539840698242, -0.086633786559104919434,
    -1.6812889575958251953, -0.68101739883422851562, -1.1742042303085327148,
    -0.92823976278305053711, 1.5825012922286987305)
  val inWArr = Array(-0.10945629328489303589, 1.7590323686599731445, -0.84798640012741088867,
    -1.965233922004699707, 1.0445169210433959961, 1.4108313322067260742,
    0.87440317869186401367, 0.88083744049072265625)
  val outWArr = Array(-1.9495558738708496094, -0.49799314141273498535, 1.9071881771087646484,
    -0.99247020483016967773, -0.24947847425937652588, 0.098931826651096343994,
    0.29085457324981689453, 1.1305880546569824219)
  val input = Tensor(Storage(inputArr.map(x => x.toFloat)))
  val target = new Table
  target.insert(Tensor(Storage(targetArr.map(x => x.toFloat))))
  target.insert(Tensor(Storage(inWArr.map(x => x.toFloat))))
  target.insert(Tensor(Storage(outWArr.map(x => x.toFloat))))

  "a SmoothL1CriterionWithWeights of object detection with sigma 2.4" should
    "generate correct loss and grad" in {
    val smcod = new SmoothL1CriterionWithWeights[Float](2.4f, 2)
    val expectedOutput = -2.2134404182434082031
    val actualOutput = smcod.forward(input, target)
    assert(abs(actualOutput - expectedOutput) < 1e-6)

    val expectedGrad = Tensor(Storage(Array(-0.0087008103728294372559, 0.43799301981925964355,
      0.73976248502731323242, -0.97521805763244628906, -0.1302922368049621582,
      0.069788061082363128662, 0.12716208398342132568,
      -0.49793213605880737305).map(x => x.toFloat)))
    val actualGrad = smcod.backward(input, target)


    expectedGrad.map(actualGrad, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  val input2 = Tensor(Storage(inputArr.map(x => x.toFloat))).resize(1, 2, 2, 2)
  val target2 = new Table
  target2.insert(Tensor(Storage(targetArr.map(x => x.toFloat))).resize(1, 2, 2, 2))
  target2.insert(Tensor(Storage(inWArr.map(x => x.toFloat))).resize(1, 2, 2, 2))
  target2.insert(Tensor(Storage(outWArr.map(x => x.toFloat))).resize(1, 2, 2, 2))

  "a SmoothL1CriterionWithWeights of object detection with sigma 2.4 and 4 dims" should
    "generate correct loss and grad" in {
    val smcod = new SmoothL1CriterionWithWeights[Float](2.4f, 2)
    val expectedOutput = -2.2134404182434082031
    val actualOutput = smcod.forward(input2, target2)
    assert(abs(actualOutput - expectedOutput) < 1e-6)

    val expectedGrad = Tensor(Storage(Array(-0.0087008103728294372559, 0.43799301981925964355,
      0.73976248502731323242, -0.97521805763244628906,
      -0.1302922368049621582, 0.069788061082363128662,
      0.12716208398342132568, -0.49793213605880737305).map(x => x.toFloat)))
      .resize(Array(1, 2, 2, 2))

    val actualGrad = smcod.backward(input, target)


    expectedGrad.map(actualGrad, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "SmoothL1CriterionWithWeights with sigma 1 and without weights" should
    "have the same result as SmoothL1Criterion" in {
    val targetNoWeight = Tensor(Storage(targetArr.map(x => x.toFloat)))
    val smcod = SmoothL1CriterionWithWeights[Float](1f, input.nElement())
    val smc = SmoothL1Criterion[Float](true)
    val out1 = smcod.forward(input, new Table().insert(targetNoWeight))
    val out2 = smc.forward(input, targetNoWeight)
    assert(abs(out1 - out2) < 1e-6)

    val smcodGrad = smcod.backward(input, new Table().insert(targetNoWeight))
    val smcGrad = smc.backward(input, targetNoWeight)
    smcodGrad.map(smcGrad, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "a SmoothL1CriterionWithWeights of object detection with num==0" should
    "generate correct loss and grad" in {
    val label = T()
    label.insert(target(1))
    val smcod = SmoothL1CriterionWithWeights[Float](1f)
    val smc = SmoothL1Criterion[Float](true)
    smcod.forward(input, label)
    smc.forward(input, target(1))

    smcod.output should be(smc.output)
    smcod.backward(input, label)
    smc.backward(input, target(1))

    smc.gradInput should be (smcod.gradInput)
  }
}
