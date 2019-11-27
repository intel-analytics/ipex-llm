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

import com.intel.analytics.bigdl.nn.MM
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.collection.mutable

@com.intel.analytics.bigdl.tags.Serial
class MMSpec extends TorchSpec {
    def randn(): Double = RandomGenerator.RNG.uniform(-10, 10)
  "A MM" should "generate correct output with no transform and no batch" in {
    torchCheck()
    val input1 = Tensor[Double](5, 3).apply1(x => randn())
    val input2 = Tensor[Double](3, 5).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](5, 5)
    gradOutput.apply1(x => randn())

    val module = new MM[Double]()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MM()\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MM, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MM" should "generate correct output with transform and no batch" in {
    torchCheck()
    val input1 = Tensor[Double](3, 5).apply1(x => randn())
    val input2 = Tensor[Double](5, 3).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](5, 5)
    gradOutput.apply1(x => randn())

    val module = new MM[Double](true, true)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MM(true, true)\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]


    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MM, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MM" should "generate correct output with no transform and batch" in {
    torchCheck()
    val input1 = Tensor[Double](4, 5, 3).apply1(x => randn())
    val input2 = Tensor[Double](4, 3, 5).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](4, 5, 5)
    gradOutput.apply1(x => randn())

    val module = new MM[Double]()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MM()\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MM, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }


  "A MM" should "generate correct output with transform and batch" in {
    torchCheck()
    val input1 = Tensor[Double](4, 3, 5).apply1(x => randn())
    val input2 = Tensor[Double](4, 5, 3).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](4, 5, 5)
    gradOutput.apply1(x => randn())

    val module = new MM[Double](true, true)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MM(true, true)\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MM, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
