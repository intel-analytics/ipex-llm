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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.{CriterionTable, MSECriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.HashMap
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class CriterionTableSpec extends TorchSpec {
    "A CriterionTable " should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](2, 2, 2).apply1(_ => Random.nextDouble())
    val input2 = Tensor[Double](2, 2, 2).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 2).apply1(e => Random.nextDouble())

    val input = T(input1, input2)
    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CriterionTable(nn.MSECriterion())\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]

    val module = new CriterionTable[Double](new MSECriterion())
    val start = System.nanoTime()
    val output = module.forward(input1, input2)
    val gradInput = module.backward(input1, input2)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be (luaOutput1)
    gradInput should be (luaOutput2[Tensor[Double]](1))

    println("Test case : CriterionTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
