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

import com.intel.analytics.bigdl.nn.{Power}
import com.intel.analytics.bigdl.tensor.Tensor

@com.intel.analytics.bigdl.tags.Serial
class PowerSpec extends TorchSpec {
    "A Power(2)" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Power[Double](2)
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 1
    input(Array(1, 1, 2)) = 2
    input(Array(1, 2, 1)) = 3
    input(Array(1, 2, 2)) = 4
    input(Array(2, 1, 1)) = 5
    input(Array(2, 1, 2)) = 6
    input(Array(2, 2, 1)) = 7
    input(Array(2, 2, 2)) = 8
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.1
    gradOutput(Array(1, 1, 2)) = 0.2
    gradOutput(Array(1, 2, 1)) = 0.3
    gradOutput(Array(1, 2, 2)) = 0.4
    gradOutput(Array(2, 1, 1)) = 0.5
    gradOutput(Array(2, 1, 2)) = 0.6
    gradOutput(Array(2, 2, 1)) = 0.7
    gradOutput(Array(2, 2, 2)) = 0.8

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Power(2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Power, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Power(3)" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Power[Double](3)
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 1
    input(Array(1, 1, 2)) = 2
    input(Array(1, 2, 1)) = 3
    input(Array(1, 2, 2)) = 4
    input(Array(2, 1, 1)) = 5
    input(Array(2, 1, 2)) = 6
    input(Array(2, 2, 1)) = 7
    input(Array(2, 2, 2)) = 8
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.1
    gradOutput(Array(1, 1, 2)) = 0.2
    gradOutput(Array(1, 2, 1)) = 0.3
    gradOutput(Array(1, 2, 2)) = 0.4
    gradOutput(Array(2, 1, 1)) = 0.5
    gradOutput(Array(2, 1, 2)) = 0.6
    gradOutput(Array(2, 2, 1)) = 0.7
    gradOutput(Array(2, 2, 2)) = 0.8

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Power(3)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Power, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
