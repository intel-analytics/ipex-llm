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

import com.intel.analytics.bigdl.nn.Squeeze
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SqueezeSpec extends TorchSpec {
    "A Squeeze(2)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Squeeze[Double](2)
    val input = Tensor[Double](2, 1, 2).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 2).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Squeeze(2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Squeeze, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Squeeze()" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Squeeze[Double]()
    val input = Tensor[Double](1, 1, 2, 2).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 2).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Squeeze()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Squeeze, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Squeeze(2, 3)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Squeeze[Double](2, 3)
    val input = Tensor[Double](1, 1, 2, 2).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 2).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Squeeze(2, 3)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Squeeze, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Squeeze(Array(2, 3), true)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Squeeze[Double](Array(2, 3), true)
    val input = Tensor[Double](2, 2, 1, 1).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](2, 2).apply1(_ => Random.nextDouble())

    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)

    output.size() should be (Array(2, 2))
    gradInput.size() should be (Array(2, 2, 1, 1))
  }
}
