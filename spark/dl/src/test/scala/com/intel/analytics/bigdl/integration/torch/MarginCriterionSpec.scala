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

import com.intel.analytics.bigdl.nn.MarginCriterion
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class MarginCriterionSpec extends TorchSpec {
    "A MarginCriterion " should "generate correct output and grad" in {
    torchCheck()
    val mse = new MarginCriterion[Double]
    val input = Tensor[Double](2, 2, 2).apply1(e => Random.nextDouble())
    val target = Tensor[Double](2, 2, 2).apply1(e => Random.nextDouble())

    val start = System.nanoTime()
    val output = mse.forward(input, target)
    val gradInput = mse.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "mse = nn.MarginCriterion()\n" +
      "output = mse:forward(input,target)\n" +
      "gradInput = mse:backward(input,target)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : MarginCriterion, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
