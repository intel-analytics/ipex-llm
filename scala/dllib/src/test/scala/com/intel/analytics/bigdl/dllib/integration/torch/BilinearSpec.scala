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
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class BilinearSpec extends TorchSpec {

  "BiLinear L2 regularizer" should "works correctly" in {
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

    val input1 = Tensor[Double](batchSize, inputN).rand()
    val input2 = Tensor[Double](batchSize, inputN).rand()
    val input = T(input1, input2)
    val labels = Tensor[Double](batchSize, outputN).rand()

    val model1 = Sequential()
      .add(Bilinear(inputN, inputN, outputN))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(Bilinear(inputN, inputN, outputN,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)))
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

  "A Bilinear " should "generate correct output and grad" in {
    torchCheck()
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
