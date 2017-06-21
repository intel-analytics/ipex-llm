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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class StrideSliceSpec extends FlatSpec with Matchers {

  "StrideSlice " should "compute correct output and gradient" in {
    val module1 = new StrideSlice[Double](Array((1, 1, 2, 1)))
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.17020166106522
    input(Array(1, 1, 2)) = 0.57785657607019
    input(Array(1, 2, 1)) = -1.3404131438583
    input(Array(1, 2, 2)) = 1.0938102817163
    input(Array(2, 1, 1)) = 1.120370157063
    input(Array(2, 1, 2)) = -1.5014141565189
    input(Array(2, 2, 1)) = 0.3380249235779
    input(Array(2, 2, 2)) = -0.625677742064

    val expectOutput1 = Tensor[Double](1, 2, 2)
    expectOutput1(Array(1, 1, 1)) = -0.17020166106522
    expectOutput1(Array(1, 1, 2)) = 0.57785657607019
    expectOutput1(Array(1, 2, 1)) = -1.3404131438583
    expectOutput1(Array(1, 2, 2)) = 1.0938102817163

    val expectedGradInput = Tensor[Double](2, 2, 2)
    expectedGradInput(Array(1, 1, 1)) = -0.17020166106522
    expectedGradInput(Array(1, 1, 2)) = 0.57785657607019
    expectedGradInput(Array(1, 2, 1)) = -1.3404131438583
    expectedGradInput(Array(1, 2, 2)) = 1.0938102817163
    expectedGradInput(Array(2, 1, 1)) = 0.0
    expectedGradInput(Array(2, 1, 2)) = 0.0
    expectedGradInput(Array(2, 2, 1)) = 0.0
    expectedGradInput(Array(2, 2, 2)) = 0.0


    val output1 = module1.forward(input)
    val gradInput = module1.backward(input, output1)

    output1 should be(expectOutput1)
    gradInput should be(expectedGradInput)
  }

}
