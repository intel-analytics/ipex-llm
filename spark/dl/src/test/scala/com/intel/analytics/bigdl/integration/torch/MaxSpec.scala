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

import com.intel.analytics.bigdl.nn.Max
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class MaxSpec extends TorchSpec {
    "A Max(2)" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Max[Double](2)
    val input = Tensor[Double](2, 3, 4).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 4).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Max(2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Max, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Max()" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Max[Double]()
    val input = Tensor[Double](1, 2, 3, 4).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 3, 4).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Max()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Max, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Max(2, 3)" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Max[Double](2, 3)
    val input = Tensor[Double](3, 5, 3, 4).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 5, 4).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Max(2, 3)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Max, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
