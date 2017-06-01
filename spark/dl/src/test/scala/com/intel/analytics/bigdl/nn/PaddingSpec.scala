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

import com.intel.analytics.bigdl.nn.Padding.PaddingInfo
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class PaddingSpec extends FlatSpec with Matchers {
  "A Padding(Array(Array(2, 3), Array(-1, 1, -2, 2))" should
    "generate correct output and grad for multi dimension" in {
    val layer = Padding[Float](
      Array(PaddingInfo(2, 1, 1, 1, 1), PaddingInfo(3, 1, 1, 1, 1)), 3, 0.0)
    val input = Tensor[Float](1, 2, 2)
    input(Array(1, 1, 1)) = 0.01f
    input(Array(1, 1, 2)) = 0.02f
    input(Array(1, 2, 1)) = 0.03f
    input(Array(1, 2, 2)) = 0.04f
    val expectedOutput = Tensor[Float](1, 4, 4)
    expectedOutput(Array(1, 1, 1)) = 0.00f
    expectedOutput(Array(1, 1, 2)) = 0.00f
    expectedOutput(Array(1, 1, 3)) = 0.00f
    expectedOutput(Array(1, 1, 4)) = 0.00f
    expectedOutput(Array(1, 2, 1)) = 0.00f
    expectedOutput(Array(1, 2, 2)) = 0.01f
    expectedOutput(Array(1, 2, 3)) = 0.02f
    expectedOutput(Array(1, 2, 4)) = 0.00f
    expectedOutput(Array(1, 3, 1)) = 0.00f
    expectedOutput(Array(1, 3, 2)) = 0.03f
    expectedOutput(Array(1, 3, 3)) = 0.04f
    expectedOutput(Array(1, 3, 4)) = 0.00f
    expectedOutput(Array(1, 4, 1)) = 0.00f
    expectedOutput(Array(1, 4, 2)) = 0.00f
    expectedOutput(Array(1, 4, 3)) = 0.00f
    expectedOutput(Array(1, 4, 4)) = 0.00f

    val gradOutput = Tensor[Float](1, 4, 4)
    gradOutput(Array(1, 1, 1)) = 0.01f
    gradOutput(Array(1, 1, 2)) = 0.02f
    gradOutput(Array(1, 1, 3)) = 0.03f
    gradOutput(Array(1, 1, 4)) = 0.04f
    gradOutput(Array(1, 2, 1)) = 0.05f
    gradOutput(Array(1, 2, 2)) = 0.06f
    gradOutput(Array(1, 2, 3)) = 0.07f
    gradOutput(Array(1, 2, 4)) = 0.08f
    gradOutput(Array(1, 3, 1)) = 0.09f
    gradOutput(Array(1, 3, 2)) = 0.10f
    gradOutput(Array(1, 3, 3)) = 0.11f
    gradOutput(Array(1, 3, 4)) = 0.12f
    gradOutput(Array(1, 4, 1)) = 0.13f
    gradOutput(Array(1, 4, 2)) = 0.14f
    gradOutput(Array(1, 4, 3)) = 0.15f
    gradOutput(Array(1, 4, 4)) = 0.16f
    val expectedGrad = Tensor[Float](1, 2, 2)
    expectedGrad(Array(1, 1, 1)) = 0.06f
    expectedGrad(Array(1, 1, 2)) = 0.07f
    expectedGrad(Array(1, 2, 1)) = 0.10f
    expectedGrad(Array(1, 2, 2)) = 0.11f

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