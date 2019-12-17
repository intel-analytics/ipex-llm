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

import com.intel.analytics.bigdl.nn.Cosine
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class CosineSpec extends TorchSpec {
    "A Cosine Module" should "generate correct output and grad in 2D" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val module = new Cosine[Double](2, 3)

    val input = Tensor[Double](3, 2).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 3).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.Cosine(2, 3)\n" +
      "gradWeight = module.gradWeight\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, gradOutput)\n"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradWeight"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)
    luagradWeight should be(gradWeight)

    println("Test case : Cosine, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")

  }

  "A Cosine Module" should "generate correct output and grad in 1D" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = new Cosine[Double](3, 2)
    var input = Tensor[Double](3).apply1(_ => Random.nextDouble())

    val gradOutput = Tensor[Double](2).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.Cosine(3, 2)\n" +
      "gradWeight = module.gradWeight\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradWeight"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)
    luagradWeight should be(gradWeight)

    println("Test case : Cosine, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
