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

import com.intel.analytics.bigdl.nn.Bilinear
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class BilinearSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A Bilinear " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 2).apply1(e => Random.nextDouble())

    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.Bilinear(5,3,2)\n" +
      "module:reset()\n" +
      "bias = module.bias\n" +
      "weight = module.weight\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n" +
      "gradBias = module.gradBias\n" +
      "gradWeight = module.gradWeight\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "bias", "weight", "grad", "gradBias", "gradWeight"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Table]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val module = new Bilinear[Double](5, 3, 2)
    val start = System.nanoTime()
    module.reset()
    val bias = module.bias
    val output = module.forward(input)
    val weight = module.weight
    val gradBias = module.gradBias
    val gradWeight = module.gradWeight
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    bias should be(luaBias)
    weight should be(luaWeight)
    gradBias should be(luaGradBias)
    gradWeight should be(luaGradWeight)

    luaOutput2 should be (gradInput)

    println("Test case : Bilinear, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
