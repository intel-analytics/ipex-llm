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
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SpatialDilatedConvolutionSpec extends TorchSpec {
  "SpatialDilatedConvolution L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    torchCheck()
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val inputN = 5
    val outputN = 2
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val labels = Tensor[Double](4).rand()

    val model1 = Sequential()
      .add(new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
        kW, kH, dW, dH, padW, padH))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
        kW, kH, dW, dH, padW, padH,
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

  "A SpatialDilatedConvolution" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kW = 3
    val kH = 3
    val dW = 1
    val dH = 1
    val padW = 2
    val padH = 2
    val layer = new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](3, 3, 6, 6).apply1(e => Random.nextDouble())
    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialDilatedConvolution(3, 6, 3, 3, 1, 1, 2, 2)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input) "

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight)
    bias should be(luaBias)
    output should be(luaOutput)
  }

  "A SpatialDilatedConvolution" should "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kW = 3
    val kH = 3
    val dW = 1
    val dH = 1
    val padW = 2
    val padH = 2
    val layer = new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, 3, 6, 6).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialDilatedConvolution(3, 6, 3, 3, 1, 1, 2, 2)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("weight", "bias", "output", "gradInput", "gradBias", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight)
    bias should be(luaBias)
    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight)
  }

  "A SpatialDilatedConvolution" should "generate correct output and grad with 3D input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kW = 3
    val kH = 3
    val dW = 2
    val dH = 2
    val padW = 1
    val padH = 1
    val layer = new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, 6, 6).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    val gradInput = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialDilatedConvolution(3, 6, 3, 3, 2, 2, 1, 1)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("weight", "bias", "output", "gradInput", "gradBias", "gradWeight")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight)
    bias should be(luaBias)
    output should be(luaOutput)
    gradInput should be(luaGradInput)
    luaGradBias should be (layer.gradBias)
    luaGradWeight should be (layer.gradWeight)
  }

  "A SpatialDilatedConvolution multiple forward backward" should
    "generate correct output and grad" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 6
    val kW = 3
    val kH = 3
    val dW = 1
    val dH = 1
    val padW = 2
    val padH = 2
    val diaW = 2
    val diaH = 2
    val layer = new SpatialDilatedConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH, diaW, diaH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](3, 3, 6, 6).apply1(e => Random.nextDouble())
    val output = model.updateOutput(input).toTensor[Double]

    val gradOutput = Tensor[Double]().resizeAs(output).apply1(e => Random.nextDouble())

    model.zeroGradParameters()
    val gradInput = model.backward(input, gradOutput)

    val output2 = model.forward(input).toTensor[Double]
    model.zeroGradParameters()
    val gradInput2 = model.backward(input, gradOutput)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialDilatedConvolution(3, 6, 3, 3, 1, 1, 2, 2, 2, 2)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input)
      gradInput = model:backward(input, gradOutput)
      gradBias = layer.gradBias
      gradWeight = layer.gradWeight
      model:zeroGradParameters()
      output2 = model:forward(input)
      gradInput2 = model:backward(input, gradOutput)
      gradBias2 = layer.gradBias
      gradWeight2 = layer.gradWeight
      """

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "gradOutput" -> gradOutput),
      Array("weight", "bias", "output", "gradInput", "gradBias", "gradWeight",
        "output2", "gradInput2", "gradBias2", "gradWeight2")
    )

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradBias = torchResult("gradBias").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaGradInput2 = torchResult("gradInput2").asInstanceOf[Tensor[Double]]
    val luaGradBias2 = torchResult("gradBias2").asInstanceOf[Tensor[Double]]
    val luaGradWeight2 = torchResult("gradWeight2").asInstanceOf[Tensor[Double]]


    val weight = layer.weight
    val bias = layer.bias

    weight should be(luaWeight)
    bias should be(luaBias)
    output2 should be(luaOutput2)
    gradInput2 should be(luaGradInput2)
    luaGradBias2 should be (layer.gradBias)
    luaGradWeight2 should be (layer.gradWeight)
  }
}
