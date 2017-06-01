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

class MeanSpec extends FlatSpec with Matchers {
  "A MeanMulDim(Array(2, 3), 3)" should "generate correct output and grad for multi dimension" in {
    val layer = Mean[Float](Array(2, 3), -1, true)
    val input = Tensor[Float](1, 3, 3)
    input(Array(1, 1, 1)) = 0.01f
    input(Array(1, 1, 2)) = 0.02f
    input(Array(1, 1, 3)) = 0.03f
    input(Array(1, 2, 1)) = 0.04f
    input(Array(1, 2, 2)) = 0.05f
    input(Array(1, 2, 3)) = 0.06f
    input(Array(1, 3, 1)) = 0.07f
    input(Array(1, 3, 2)) = 0.08f
    input(Array(1, 3, 3)) = 0.09f
    val gradOutput = Tensor[Float](1, 1, 1)
    gradOutput(Array(1, 1, 1)) = 0.09f
    val expectedOutput = Tensor[Float](1, 1, 1)
    expectedOutput(Array(1, 1, 1)) = 0.05f
    val expectedGrad = Tensor[Float](1, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0.01f
    expectedGrad(Array(1, 1, 2)) = 0.01f
    expectedGrad(Array(1, 1, 3)) = 0.01f
    expectedGrad(Array(1, 2, 1)) = 0.01f
    expectedGrad(Array(1, 2, 2)) = 0.01f
    expectedGrad(Array(1, 2, 3)) = 0.01f
    expectedGrad(Array(1, 3, 1)) = 0.01f
    expectedGrad(Array(1, 3, 2)) = 0.01f
    expectedGrad(Array(1, 3, 3)) = 0.01f
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)

    expectedOutput should be (output)
    expectedGrad should be (gradInput)
    input should be (inputOrg)
    gradOutput should be (gradOutputOrg)
  }
}