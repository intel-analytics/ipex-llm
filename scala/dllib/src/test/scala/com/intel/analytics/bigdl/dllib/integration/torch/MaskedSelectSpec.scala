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

import com.intel.analytics.bigdl.nn.MaskedSelect
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class MaskedSelectSpec extends TorchSpec {
    "A MaskedSelect Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new MaskedSelect[Double]()
    val input1 = Tensor[Double](2, 2).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](2, 2)
    input2(Array(1, 1)) = 1
    input2(Array(1, 2)) = 0
    input2(Array(2, 1)) = 0
    input2(Array(2, 2)) = 1

    val input = new Table()
    input(1.0) = input1
    input(2.0) = input2

    val gradOutput = Tensor[Double](5).apply1(e => Random.nextDouble())

    val code = "module = nn.MaskedSelect()\n" +
      "mask = torch.ByteTensor({{1, 0}, {0, 1}})\n" +
      "output = module:forward({input1, mask})\n" +
      "gradInput = module:backward({input1, mask}, gradOutput)\n" +
      "gradInput[2] = gradInput[2]:double()"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be (luaOutput1)
    gradInput should equal (luaOutput2)

    println("Test case : MaskedSelect, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
