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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{GradientChecker, LogSoftMax}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class LogSoftMaxSpec extends TorchSpec {
    "A LogSoftMax Module " should "generate correct output and grad with input 2D" in {
    torchCheck()
    val module = new LogSoftMax[Double]()
    Random.setSeed(100)
    val input = Tensor[Double](4, 10).apply1(e => Random.nextDouble())
    val data = Tensor[Double](4, 20).randn()
    val gradOutput = data.narrow(2, 1, 10)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.LogSoftMax()\n" +
      "output1 = module:forward(input)\n " +
      "output2 = module:backward(input, gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output1", "output2"))
    val luaOutput = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : LogSoft, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LogSoftMax Module " should "generate correct output and grad with input 1D" in {
    torchCheck()
    val module = new LogSoftMax[Double]()
    Random.setSeed(100)
    val input = Tensor[Double](10).apply1(e => Random.nextDouble())
    val data = Tensor[Double](20).randn()
    val gradOutput = data.narrow(1, 1, 10)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.LogSoftMax()\n" +
      "output1 = module:forward(input)\n " +
      "output2 = module:backward(input, gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output1", "output2"))
    val luaOutput = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : LogSoft, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A LogSoftMax Module " should "generate correct output and grad tiwh input 1*N" in {
    torchCheck()
    val module = new LogSoftMax[Double]()
    Random.setSeed(100)
    val input = Tensor[Double](1, 10).apply1(e => Random.nextDouble())
    val data = Tensor[Double](1, 20).randn()
    val gradOutput = data.narrow(2, 1, 10)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.LogSoftMax()\n" +
      "output1 = module:forward(input)\n " +
      "output2 = module:backward(input, gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output1", "output2"))
    val luaOutput = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : LogSoft, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "LogSoftMax module" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val layer = new LogSoftMax[Double]()
    val input = Tensor[Double](4, 10).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

  "LogSoftMax float module" should "return good result" in {
    torchCheck()
    val module = new LogSoftMax[Float]()
    Random.setSeed(100)
    val input = Tensor[Float](2, 5).apply1(e => Random.nextFloat() + 10)
    val gradOutput = Tensor[Float](2, 5).apply1(e => Random.nextFloat() + 10)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.setdefaulttensortype('torch.FloatTensor')" +
      "module = nn.LogSoftMax()\n" +
      "output1 = module:forward(input)\n " +
      "output2 = module:backward(input, gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output1", "output2"))
    val luaOutput = torchResult("output1").asInstanceOf[Tensor[Float]]
    val luaGradInput = torchResult("output2").asInstanceOf[Tensor[Float]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

  }
}
