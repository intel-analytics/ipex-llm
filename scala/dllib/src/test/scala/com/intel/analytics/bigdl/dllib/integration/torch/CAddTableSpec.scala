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

import com.intel.analytics.bigdl.nn.{CAddTable, ConcatTable, Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class CAddTableSpec extends TorchSpec {
    "CAddTable with ConcatTable" should "return right output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val model = new Sequential[Double]()
    val ctable = new ConcatTable[Double]()
    ctable.add(new Linear(5, 3))
    ctable.add(new Linear(5, 3))
    model.add(ctable)
    model.add(CAddTable())
    val input = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](3).apply1(_ => Random.nextDouble())

    val output = model.forward(input)
    val gradInput = model.updateGradInput(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """model = nn.Sequential()
         ctable = nn.ConcatTable():add(nn.Linear(5, 3)):add(nn.Linear(5, 3))
         model:add(ctable)
         model:add(nn.CAddTable())
        output = model:forward(input)
        gradInput = model:backward(input, gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
  }

  "CAddTable inplace with ConcatTable" should "return right output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val model = new Sequential[Double]()
    val ctable = new ConcatTable[Double]()
    ctable.add(new Linear(5, 3))
    ctable.add(new Linear(5, 3))
    model.add(ctable)
    model.add(CAddTable(true))
    val input = Tensor[Double](5).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](3).apply1(_ => Random.nextDouble())

    val output = model.forward(input)
    val gradInput = model.updateGradInput(input, gradOutput)
    model.accGradParameters(input, gradOutput)


    val code = "torch.manualSeed(" + seed + ")\n" +
      """model = nn.Sequential()
         ctable = nn.ConcatTable():add(nn.Linear(5, 3)):add(nn.Linear(5, 3))
         model:add(ctable)
         model:add(nn.CAddTable(true))
        output = model:forward(input)
        gradInput = model:backward(input, gradOutput)
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
  }

}
