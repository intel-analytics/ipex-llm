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

import com.intel.analytics.bigdl.nn.{LeakyReLU, RReLU}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.utils.RandomGenerator._

@com.intel.analytics.bigdl.tags.Serial
class LeakyReLUSpec  extends TorchSpec {
    def random(): Double = RandomGenerator.RNG.normal(-10, 10)

  "A LeakyReLU Module" should "generate correct output and grad not inplace" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = new LeakyReLU[Double]()
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => random())
    val gradOutput = Tensor[Double](2, 2, 2)
    input.apply1(x => random())

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LeakyReLU()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : LeakyReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LeakyReLU Module" should "generate correct output and grad inplace" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = LeakyReLU[Double](inplace = true)
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => random())
    val gradOutput = Tensor[Double](2, 2, 2)
    input.apply1(x => random())

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LeakyReLU(1/100,true)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input.clone(), gradOutput.clone())
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : LeakyReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LeakyReLU Float Module" should "generate correct output and grad not inplace" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = new LeakyReLU[Float]()
    val input = Tensor[Float](2, 2, 2)
    input.apply1(x => random().toFloat)
    val gradOutput = Tensor[Float](2, 2, 2)
    input.apply1(x => random().toFloat)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')\n" +
      "module = nn.LeakyReLU()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Float]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Float]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : LeakyReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LeakyReLU Float Module" should "generate correct output and grad inplace" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = LeakyReLU[Float](inplace = true)
    val input = Tensor[Float](2, 2, 2)
    input.apply1(x => random().toFloat)
    val gradOutput = Tensor[Float](2, 2, 2)
    input.apply1(x => random().toFloat)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')\n" +
      "module = nn.LeakyReLU(1/100,true)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Float]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Float]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input.clone(), gradOutput.clone())
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : LeakyReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LeakyReLU Float Module incontiguous input" should
    "generate correct output and grad not inplace" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = new LeakyReLU[Float]()
    val input = Tensor[Float](4, 2, 2).narrow(1, 2, 2)
    input.apply1(x => random().toFloat)
    val gradOutput = Tensor[Float](4, 2, 2).narrow(1, 2, 2)
    input.apply1(x => random().toFloat)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')\n" +
      "module = nn.LeakyReLU()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Float]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Float]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : LeakyReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LeakyReLU Float Module incontiguous input" should
    "generate correct output and grad inplace" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val module = LeakyReLU[Float](inplace = true)
    val input = Tensor[Float](4, 2, 2).narrow(1, 2, 2)
    input.apply1(x => random().toFloat)
    val gradOutput = Tensor[Float](4, 2, 2).narrow(1, 2, 2)
    input.apply1(x => random().toFloat)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "torch.setdefaulttensortype('torch.FloatTensor')\n" +
      "module = nn.LeakyReLU(1/100,true)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Float]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Float]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input.clone(), gradOutput.clone())
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output)
    luaOutput2 should be (gradInput)

    println("Test case : LeakyReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
