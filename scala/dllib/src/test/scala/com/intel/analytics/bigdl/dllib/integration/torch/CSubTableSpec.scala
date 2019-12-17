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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.nn.CSubTable

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class CSubTableSpec extends TorchSpec {
    "A CDivTable Module" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val module = new CSubTable[Double]()

    val input1 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CSubTable()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    luaOutput1 should be(output)
    luaOutput2 should be (gradInput)

    println("Test case : CSubTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
