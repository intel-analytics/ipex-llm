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

package com.intel.analytics.bigdl.dllib.integration.torch

import com.intel.analytics.bigdl.dllib.nn.Dropout
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._

@com.intel.analytics.bigdl.tags.Serial
class DropoutSpec extends TorchSpec {
    "Dropout module with continuous input" should "converge to correct weight and bias" in {
    torchCheck()
    val module = new Dropout[Double](0.7, false, true)
    val input = Tensor[Double](100, 100)
    val seed = 100

    input.rand()

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.Dropout(0.7)\n" +
      "output1 = module:forward(input)\n" +
      "output2 = module:backward(input, input:clone():fill(1))"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    val output1 = module.forward(input)
    val output2 = module.backward(input, input.clone().fill(1))
    val end = System.nanoTime()
    val scalaTime = end - start


    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : Dropout, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Dropout module with discontinuous input" should "converge to correct weight and bias" in {
    torchCheck()
    val module = new Dropout[Double](0.7, false, true)
    val input = Tensor[Double](100, 100).t
    val seed = 100

    input.rand()

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.Dropout(0.7)\n" +
      "output1 = module:forward(input)\n" +
      "output2 = module:backward(input, input:clone():fill(1))"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    val output1 = module.forward(input)
    val output2 = module.backward(input, input.clone().fill(1))
    val end = System.nanoTime()
    val scalaTime = end - start


    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : Dropout, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Dropout performance" should "work" in {
    val module = new Dropout[Float](0.5, true, false)
    val input = Tensor[Float](1, 12, 512, 512)
    val seed = 100
    val start = System.nanoTime()
    input.bernoulli(0.5)
    println((System.nanoTime() - start) / 1e9)
    input.randn()
    val start2 = System.nanoTime()
    val output = module.forward(input)
    println((System.nanoTime() - start2) / 1e9)
  }
  }
