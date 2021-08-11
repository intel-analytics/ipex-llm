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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class MultiCriterionSpec extends TorchSpec {
    "A MultiCriterion Module " should "generate correct output and grad with Tensor input" in {
    torchCheck()
    RNG.setSeed(10)
    val module = new MultiCriterion[Double]()
    val nll = new ClassNLLCriterion[Double]()
    val nll2 = new MSECriterion[Double]()
    module.add(nll, 0.5)
    module.add(nll2)

    val input = Tensor[Double](5).rand()
    val target = Tensor[Double](5)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    target(Array(4)) = 2
    target(Array(5)) = 1

    val code = "nll = nn.ClassNLLCriterion()\n" +
      "nll2 = nn.MSECriterion()\n" +
      "module = nn.MultiCriterion():add(nll, 0.5):add(nll2)\n" +
      "output = module:forward(input, target)\n" +
      "gradInput = module:backward(input, target)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val output = module.forward(input, target)
    val gradInput = module.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : MultiCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MultiCriterion Module " should "generate correct output and grad with Table input" in {
    torchCheck()
    // todo: need to check table input
  }
}
