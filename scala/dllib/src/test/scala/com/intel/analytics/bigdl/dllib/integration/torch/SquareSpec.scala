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

import com.intel.analytics.bigdl.nn.Square
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SquareSpec extends TorchSpec {
    "A Square 1D input" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Square[Double]()
    val input = Tensor[Double](10)
    input.apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](10)
    gradOutput.apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Square()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Square, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Square 2D input" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Square[Double]()
    val input = Tensor[Double](3, 5)
    input.apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 5)
    gradOutput.apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Square()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Square, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Square 3D input" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Square[Double]()
    val input = Tensor[Double](4, 6, 6)
    input.apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](4, 6, 6)
    gradOutput.apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Square()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Square, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Square 4D input" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Square[Double]()
    val input = Tensor[Double](3, 5, 6, 6)
    input.apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 5, 6, 6)
    gradOutput.apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Square()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Square, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
