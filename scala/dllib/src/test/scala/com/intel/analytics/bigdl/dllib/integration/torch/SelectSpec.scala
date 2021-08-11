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

import com.intel.analytics.bigdl.nn.Select
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

@com.intel.analytics.bigdl.tags.Serial
class SelectSpec extends TorchSpec {
    "Select(3, 5)" should "generate correct output and grad" in {
    torchCheck()
    def randn(): Double = RandomGenerator.RNG.uniform(-10, 10)
    val layer = new Select[Double](3, 5)
    val input = Tensor[Double](5, 5, 5)
    input.apply1(x => randn())
    val gradOutput = Tensor[Double](5, 5, 1)
    gradOutput.apply1(x => randn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Select(3, 5)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Select, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Select(2, 5)" should "generate correct output and grad" in {
      torchCheck()
      def randn(): Double = RandomGenerator.RNG.uniform(-10, 10)
      val layer = new Select[Double](2, 5)
      val input = Tensor[Double](3, 5, 5)
      input.apply1(x => randn())
      val gradOutput = Tensor[Double](3, 5, 1)
      gradOutput.apply1(x => randn())

      val start = System.nanoTime()
      val output = layer.forward(input)
      val gradInput = layer.backward(input, gradOutput)
      val end = System.nanoTime()
      val scalaTime = end - start

      val code = "module = nn.Select(2, 5)\n" +
        "output = module:forward(input)\n" +
        "gradInput = module:backward(input,gradOutput)"

      val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
          Array("output", "gradInput"))
      val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
      val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

      output should be (luaOutput)
      gradInput should be (luaGradInput)

      println("Test case : Select, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
