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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T

@com.intel.analytics.bigdl.tags.Serial
class LookupTableSpec extends TorchSpec {

  "LookupTable L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    torchCheck()

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val inputN = 5
    val outputN = 2
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN).apply1(x => RNG.uniform(1, inputN))
    val labels = Tensor[Double](batchSize, inputN, outputN).rand()

    val model1 = Sequential()
      .add(LookupTable(inputN, outputN))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(LookupTable(inputN, outputN,
        wRegularizer = L2Regularizer(0.1)))
      .add(Sigmoid())
    val (weights2, grad2) = model2.getParameters()
    weights2.copy(weights1.clone())
    grad2.copy(grad1.clone())


    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss1: Array[Double] = null
    for (i <- 1 to 100) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
      println(s"${i}-th loss = ${loss1(0)}")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${i}-th loss = ${loss2(0)}")
    }


    weights1 should be(weights2)
    loss1 should be(loss2)
  }

  "A LookupTableSpec" should "generate correct output and grad with input 1D" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](9, 4, 2, 0.1, 2.0, true)
    val input = Tensor[Double](5)
    input(Array(1)) = 5
    input(Array(2)) = 2
    input(Array(3)) = 6
    input(Array(4)) = 9
    input(Array(5)) = 4

    val gradOutput = Tensor[Double](5, 4).rand()

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LookupTable(9, 4, 2, 0.1)\n" +
      "module:scaleGradByFreq()\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input:int())\n" +
      "module._count:zero()\n" +
      "_gradInput = module:backward(input:int(), gradOutput)\n" +
      "i = i + 1\n" +
      "end\n" +
      "gradInput = _gradInput:double()\n" +
      "weight = module.weight\n" +
      "gradweight = module.gradWeight\n" +
      "count = module._count:double()\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "weight", "gradweight", "count"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaweight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradweight").asInstanceOf[Tensor[Double]]
    val luaCount = torchResult("count").asInstanceOf[Tensor[Int]]

    val start = System.nanoTime()
    var output : Tensor[Double] = null
    var gradInput : Tensor[Double] = null
    var i = 0
    while (i < 10) {
      output = module.forward(input)
      gradInput = module.backward(input, gradOutput)
      i += 1
    }
    val weight = module.weight
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    luaweight should be(weight)
    luagradWeight should be (gradWeight)
    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : LookupTable, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A LookupTableSpec" should "generate correct output and grad with input 2D" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](10, 3, 3)
    val input = Tensor[Double](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 10

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LookupTable(10, 3, 3)\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input)\n" +
      "_gradInput = module:backward(input, output)\n" +
      "i = i + 1\n" +
      "end\n" +
      "gradInput = _gradInput:double()\n" +
      "weight = module.weight\n" +
      "gradweight = module.gradWeight\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput", "weight", "gradweight", "shouldScaleGradByFreq"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaweight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradweight").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    var output : Tensor[Double] = null
    var gradInput : Tensor[Double] = null
    var i = 0
    while (i < 10) {
      output = module.updateOutput(input)
      gradInput = module.backward(input, output)
      i += 1
    }
    val weight = module.weight
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    luaweight should be(weight)
    luagradWeight should be (gradWeight)
    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : LookupTable, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A LookupTableSpec" should "generate correct output and grad with max-norm regularization" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](10, 3, 0, 0.1, 2)
    val input = Tensor[Double](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 10

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LookupTable(10, 3, 0, 0.1, 2)\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input:int())\n" +
      "_gradInput = module:backward(input:int(),output) \n" +
      "i = i + 1\n" +
      "end\n" +
      "gradInput = _gradInput:double()\n" +
      "weight = module.weight\n" +
      "gradweight = module.gradWeight\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput", "weight", "gradweight"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaweight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradweight").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    var output : Tensor[Double] = null
    var gradInput : Tensor[Double] = null
    var i = 0
    while (i < 10) {
      output = module.forward(input)
      gradInput = module.backward(input, output)
      i += 1
    }
    val weight = module.weight
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    luaweight should be(weight)
    luagradWeight should be (gradWeight)
    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : LookupTable, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }
}
