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
class CAddSpec extends TorchSpec {
  "CAdd L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val inputN = 5
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](batchSize, inputN).apply1(x => RNG.uniform(1, inputN))
    val labels = Tensor[Double](batchSize, inputN).rand()

    val model1 = Sequential()
      .add(CAdd[Double](Array(inputN)))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(CAdd[Double](Array(inputN),
        bRegularizer = L2Regularizer(0.1)))
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
      println(s"${ i }-th loss = ${ loss1(0) }")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${ i }-th loss = ${ loss2(0) }")
    }


    weights1 should be(weights2)
    loss1 should be(loss2)
  }
  "A CAdd(5, 1)" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(5, 1))
    val input = Tensor[Double](5, 4)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](5, 4)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(5, 1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A CAdd(3)" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(3))
    val input = Tensor[Double](2, 3)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](2, 3)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(3)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A CAdd(3, 4)" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(3, 4))
    val input = Tensor[Double](2, 3, 4)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](2, 3, 4)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(3, 4)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A CAdd(1, 10, 1, 1)" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Double](Array(1, 10, 1, 1))
    val input = Tensor[Double](3, 10, 500, 600)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](3, 10, 500, 600)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CAdd(1, 10, 1, 1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)" +
      "gradBias = module.gradBias"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradBias"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)
    layer.gradBias should be (luaGradBias)

    println("Test case : CAdd, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
