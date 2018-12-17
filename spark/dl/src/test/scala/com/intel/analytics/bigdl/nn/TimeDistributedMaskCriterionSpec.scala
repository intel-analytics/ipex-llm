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

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

import scala.math._

class TimeDistributedMaskCriterionSpec extends FlatSpec with Matchers {
  "TimeDistributedMaskCriterion" should "works correctly" in {
    val criterion = ClassNLLCriterion[Double](paddingValue = 0)
    val layer = TimeDistributedMaskCriterion[Double](criterion, paddingValue = 0)

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404

    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 0
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 0
    target(Array(3, 2)) = 3

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    val expectedOutput = 1.25822551560405
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -0.25
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -0.25
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -0.25
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = 0
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -0.25
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }
}
