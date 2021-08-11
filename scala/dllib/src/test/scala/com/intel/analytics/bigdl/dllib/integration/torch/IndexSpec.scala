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

import com.intel.analytics.bigdl.nn.Index
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class IndexSpec extends TorchSpec {
    "A Index " should "generate correct output and grad with one dimension" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](3).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](4)
    input2(Array(1)) = 1
    input2(Array(2)) = 2
    input2(Array(3)) = 2
    input2(Array(4)) = 3
    val gradOutput = Tensor[Double](4).apply1(e => Random.nextDouble())

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val code = "torch.manualSeed(" + seed + ")\n" +
      "input = {input1, torch.LongTensor{1, 2, 2, 3}}\n" +
      "module = nn.Index(1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val module = new Index[Double](1)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    luaOutput2 should be (gradInput)

    println("Test case : Index, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Index " should "generate correct output and grad with two dimension" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](3, 3).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](4)
    input2(Array(1)) = 1
    input2(Array(2)) = 2
    input2(Array(3)) = 3
    input2(Array(4)) = 1

    val gradOutput = Tensor[Double](3, 4).apply1(e => Random.nextDouble())

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val code = "torch.manualSeed(" + seed + ")\n" +
      "input = {input1, torch.LongTensor{1, 2, 3, 1}}\n" +
      "module = nn.Index(2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val module = new Index[Double](2)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    luaOutput2 should be (gradInput)

    println("Test case : Index, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }


}
