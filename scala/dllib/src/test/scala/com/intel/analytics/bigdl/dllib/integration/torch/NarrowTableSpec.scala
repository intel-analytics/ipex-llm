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

import com.intel.analytics.bigdl.nn.NarrowTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class NarrowTableSpec extends TorchSpec {
    "A NarrowTable Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new NarrowTable[Double](1, 2)

    val input = T()
    input(1.0) = Tensor[Double](2, 3).apply1(e => Random.nextDouble())
    input(2.0) = Tensor[Double](2, 1).apply1(e => Random.nextDouble())
    input(3.0) = Tensor[Double](2, 2).apply1(e => Random.nextDouble())

    val gradOutput = T()
    gradOutput(1.0) = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    gradOutput(2.0) = Tensor[Double](2, 5).apply1(e => Random.nextDouble())

    val code = "module = nn.NarrowTable(1, 2)\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)\n" +
      "i = i + 1\n" +
      "end"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Table]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val start = System.nanoTime()
    var i = 0
    var output = T()
    var gradInput = T()
    while (i < 10) {
      output = module.forward(input)
      gradInput = module.backward(input, gradOutput)
      i += 1
    }
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : NarrowTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A NarrowTable Module with negative length" should "generate correct output and grad" in {
    torchCheck()
    val module = new NarrowTable[Double](2, -2)

    val input = T()
    input(1.0) = Tensor[Double](2, 3).apply1(e => Random.nextDouble())
    input(2.0) = Tensor[Double](2, 1).apply1(e => Random.nextDouble())
    input(3.0) = Tensor[Double](2, 2).apply1(e => Random.nextDouble())
    input(4.0) = Tensor[Double](2, 2).apply1(e => Random.nextDouble())

    val gradOutput = T()
    gradOutput(1.0) = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    gradOutput(2.0) = Tensor[Double](2, 5).apply1(e => Random.nextDouble())

    val start = System.nanoTime()
    var i = 0
    var output = T()
    var gradInput = T()
    output = module.forward(input)
    gradInput = module.backward(input, gradOutput)
    i += 1
    val end = System.nanoTime()
    val scalaTime = end - start

    val gradInput1 = gradInput[Tensor[Double]](2.0)
    val gradInput2 = gradInput[Tensor[Double]](3.0)
    val expectedGradInput1 = gradOutput[Tensor[Double]](1.0)
    val expectedGradInput2 = gradOutput[Tensor[Double]](2.0)

    val output1 = output[Tensor[Double]](1.0)
    val output2 = output[Tensor[Double]](2.0)
    val expectedOutput1 = input[Tensor[Double]](2.0)
    val expectedOutput2 = input[Tensor[Double]](3.0)

    output1 should be (expectedOutput1)
    output2 should be (expectedOutput2)

    gradInput1 should be (expectedGradInput1)
    gradInput2 should be (expectedGradInput2)
  }
}
