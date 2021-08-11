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

package com.intel.analytics.bigdl.integration.torch

import com.intel.analytics.bigdl.nn.{KLDCriterion, GaussianSampler}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class GaussianSamplerSpec extends FlatSpec with Matchers{
  "A Sampler Module " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = GaussianSampler[Float]()

    RNG.setSeed(seed)
    val input1 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input2 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input = T(input1, input2)

    val gradOutput = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    val outputTarget = Tensor(Array(0.05043915f, 0.4935567f, 1.3664707f,
      -0.54287064f, 0.7525101f, 1.8190227f), Array(2, 3))
    val gardTarget1 = Tensor(Array(0.67074907f, 0.21010774f, 0.82585275f,
      0.4527399f, 0.13670659f, 0.87014264f), Array(2, 3))

    val gardTarget2 = Tensor(Array(-0.16532817f, -0.018657455f, 0.4493057f,
    -0.21616453f, 0.022419363f, 0.5623907f), Array(2, 3))

    output should be(outputTarget)
    gradInput[Tensor[Float]](1) should be(gardTarget1)
    gradInput[Tensor[Float]](2) should be(gardTarget2)
  }
}
