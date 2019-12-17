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

import com.intel.analytics.bigdl.nn.{GradientChecker, Linear, MSECriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random
import com.intel.analytics.bigdl._

@com.intel.analytics.bigdl.tags.Serial
class LinearSpec extends TorchSpec {
    "Linear module" should "converge to correct weight and bias" in {
    torchCheck()
    val inputN = 5
    val outputN = 2

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](inputN)
    val res = Tensor[Double](outputN)
    val grad = Tensor[Double](outputN).rand()
    val seed = 100

    input.rand()

    val code = "torch.manualSeed(" + seed + ")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear" -> linear,
      "input" -> input, "grad" -> grad),
      Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)
    luaWeight should be(weight)
    luaBias should be(bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Linear module without bias" should "converate to correct weight and bias" in {
    torchCheck()
    val inputN = 5
    val outputN = 2

    val linear = new Linear[Double](inputN, outputN, withBias = false)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](inputN)
    val res = Tensor[Double](outputN)
    val grad = Tensor[Double](outputN).rand()
    val seed = 100

    input.rand()

    val code = "torch.manualSeed(" + seed + ")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear" -> linear,
      "input" -> input, "grad" -> grad),
      Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)
    luaWeight should be(weight)
    luaBias should be(bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Linear (1024, 1000)" should "converate to correct weight and bias" in {
    torchCheck()
    val inputN = 1024
    val outputN = 1000

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](inputN).rand()
    val grad = Tensor[Double](outputN).rand()
    val seed = 100

    val code = "torch.manualSeed(" + seed + ")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear" -> linear,
      "input" -> input, "grad" -> grad),
      Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)
    luaWeight should be(weight)
    luaBias should be(bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Linear (27, 64)" should "converge to correct weight and bias" in {
    torchCheck()
    val inputN = 27
    val outputN = 64

    val linear = new Linear[Double](inputN, outputN)

    val input = Tensor[Double](1156, inputN).rand()
    val grad = Tensor[Double](1156, outputN).rand()
    val seed = 100

    val code = "torch.manualSeed(" + seed + ")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear" -> linear,
      "input" -> input, "grad" -> grad),
      Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)
    luaWeight should be(weight)
    luaBias should be(bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "Linear module" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val linear = new Linear[Double](5, 2)
    val input = Tensor[Double](3, 5).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](linear, input, 1e-3) should be(true)
  }

  "Linear module" should "be good in gradient check for weight" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val linear = new Linear[Double](5, 2)
    val input = Tensor[Double](3, 5).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](linear, input, 1e-3) should be(true)
  }

  "Linear (27, 64) without bias" should "converate to correct weight and bias" in {
    torchCheck()
    val inputN = 27
    val outputN = 64

    val linear = new Linear[Double](inputN, outputN, withBias = false)

    val input = Tensor[Double](1156, inputN).rand()
    val grad = Tensor[Double](1156, outputN).rand()
    val seed = 100

    val code = "torch.manualSeed(" + seed + ")\n" +
      "linear:reset()\n" +
      "output = linear:forward(input)\n" +
      "gradInput = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear" -> linear,
      "input" -> input, "grad" -> grad),
      Array("linear", "output", "gradInput"))
    val torchLinear = torchResult("linear").asInstanceOf[Linear[Double]]
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaWeight = torchLinear.weight
    val luaBias = torchLinear.bias

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)
    luaWeight should be(weight)
    luaBias should be(bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
