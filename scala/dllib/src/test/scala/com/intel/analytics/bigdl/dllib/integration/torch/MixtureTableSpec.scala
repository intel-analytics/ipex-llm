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

import com.intel.analytics.bigdl.nn.MixtureTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class MixtureTableSpec extends TorchSpec {
    "A MixtureTable " should "generate correct output and grad with table expertInput" in {
    torchCheck()
    val mse = new MixtureTable[Double]

    val expertInput = Tensor[Double](5, 3, 6).apply1(e => Random.nextDouble())
    val expertTable = new Table()
    expertTable(1.0) = Tensor[Double](5, 6).apply1(e => Random.nextDouble())
    expertTable(2.0) = Tensor[Double](5, 6).apply1(e => Random.nextDouble())
    expertTable(3.0) = Tensor[Double](5, 6).apply1(e => Random.nextDouble())

    val input1 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 6).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = expertTable

    val start = System.nanoTime()
    val output = mse.forward(input)
    val gradInput = mse.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "mse = nn.MixtureTable()\n" +
      "input = {input1, expertTable}\n" +
      "output = mse:forward(input)\n" +
      "gradInput = mse:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "expertTable" -> expertTable,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    output should be (luaOutput1)
    luaOutput2 should be (gradInput)

    println("Test case : MixtureTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MixtureTable " should "generate correct output and grad with tensor expertInput" in {
    torchCheck()
    val mse = new MixtureTable[Double]

    val expertInput = Tensor[Double](5, 3, 6).apply1(e => Random.nextDouble())
    val input1 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 6).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = expertInput

    val code = "mse = nn.MixtureTable()\n" +
      "output = mse:forward(input)\n" +
      "gradInput = mse:backward(input,gradOutput)\n" +
      "size = mse.size\n" +
      "dim = mse.dim"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "size", "dim"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val start = System.nanoTime()
    val output = mse.forward(input)
    val gradInput = mse.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be (luaOutput1)
    gradInput should be (luaOutput2)

    println("Test case : MixtureTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
