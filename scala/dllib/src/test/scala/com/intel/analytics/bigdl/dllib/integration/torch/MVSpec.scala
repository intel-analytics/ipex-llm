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

import com.intel.analytics.bigdl.nn.MV
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.collection.mutable

@com.intel.analytics.bigdl.tags.Serial
class MVSpec extends TorchSpec {
    def randn(): Double = RandomGenerator.RNG.uniform(-10, 10)
  "A MV" should "generate correct output with no transform and no batch" in {
    torchCheck()
    val input1 = Tensor[Double](3, 3).apply1(x => randn())
    val input2 = Tensor[Double](3).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](3)
    gradOutput.apply1(x => randn())

    val module = new MV[Double]()

    val start = System.nanoTime()
    var output = Tensor[Double]()
    var gradInput = T()

    for (i <- 1 to 5) {
      output = module.forward(input)
      gradInput = module.backward(input, gradOutput)
    }
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MV()\n" +
      "for i = 1,5,1 do\n" +
        "output = module:forward(input)\n " +
        "gradInput = module:backward(input, gradOutput)\n" +
      "end"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MV, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MV" should "generate correct output with transform and no batch" in {
    torchCheck()
    val input1 = Tensor[Double](3, 3).apply1(x => randn())
    val input2 = Tensor[Double](3).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](3)
    gradOutput.apply1(x => randn())

    val module = new MV[Double](true)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MV(true)\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MV, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MV" should "generate correct output with no transform and batch" in {
    torchCheck()
    val input1 = Tensor[Double](3, 3, 3).apply1(x => randn())
    val input2 = Tensor[Double](3, 3).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](3, 3)
    gradOutput.apply1(x => randn())

    val module = new MV[Double]()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MV()\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MV, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }


  "A MV" should "generate correct output with transform and batch" in {
    torchCheck()
    val input1 = Tensor[Double](3, 3, 3).apply1(x => randn())
    val input2 = Tensor[Double](3, 3).apply1(x => randn())
    val input = T(input1, input2)
    val gradOutput = Tensor[Double](3, 3)
    gradOutput.apply1(x => randn())

    val module = new MV[Double](true)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.MV(true)\n" +
      "output = module:forward(input)\n " +
      "gradInput = module:backward(input, gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Table]

    luaOutput should be (output)
    gradInput should equal (luaGradInput)

    println("Test case : MV, Torch : " +
      luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
