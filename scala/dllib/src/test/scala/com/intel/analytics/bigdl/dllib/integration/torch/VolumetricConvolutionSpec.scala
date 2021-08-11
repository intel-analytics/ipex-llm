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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T, TestUtils}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class VolumetricConvolutionSpec extends TorchSpec {
    "A VolumetricConvolution" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = 3
    val to = 2
    val kt = 2
    val ki = 2
    val kj = 2
    val st = 2
    val si = 2
    val sj = 2
    val padT = 1
    val padW = 1
    val padH = 1
    val outt = 6
    val outi = 6
    val outj = 6
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](3, 100, 56, 56).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
      "weight = layer.weight\n" +
      "bias = layer.bias \n" +
      "output = layer:forward(input)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("weight", "bias", "output"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be (luaWeight)
    bias should be (luaBias)
    output should be (luaOutput)
  }

  "A VolumetricConvolution without bias" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = 3
    val to = 2
    val kt = 2
    val ki = 2
    val kj = 2
    val st = 2
    val si = 2
    val sj = 2
    val padT = 1
    val padW = 1
    val padH = 1
    val outt = 6
    val outi = 6
    val outj = 6
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH, withBias = false)

    val input = Tensor[Double](3, 100, 56, 56).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH):noBias()\n" +
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

    weight should be (luaWeight)
    bias should be (luaBias)
    output should be (luaOutput)
  }


  "A VolumetricConvolution with batch input" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = 3
    val to = 2
    val kt = 2
    val ki = 2
    val kj = 2
    val st = 2
    val si = 2
    val sj = 2
    val padT = 1
    val padW = 1
    val padH = 1
    val outt = 6
    val outi = 6
    val outj = 6
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val batch = 3
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](batch, from, int, inj, ini).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH)\n" +
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

    weight should be (luaWeight)
    bias should be (luaBias)
    output shouldEqual luaOutput
  }

  "A VolumetricConvolution with batch input no bias" should "generate correct output" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = 3
    val to = 2
    val kt = 2
    val ki = 2
    val kj = 2
    val st = 2
    val si = 2
    val sj = 2
    val padT = 1
    val padW = 1
    val padH = 1
    val outt = 6
    val outi = 6
    val outj = 6
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val batch = 3
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH, withBias = false)

    val input = Tensor[Double](batch, from, int, inj, ini).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      s"layer = nn.VolumetricConvolution($from, $to, $kt, $ki, $kj, $st, $si, $sj, $padT," +
      s" $padW, $padH):noBias()\n" +
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

    weight should be (luaWeight)
    bias should be (luaBias)
    output should be (luaOutput)
  }

  "A VolumetricConvolution" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

  "A VolumetricConvolution with batch" should "be good in gradient check for input" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 6).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](layer, input, 1e-3) should be(true)
  }

  "A VolumetricConvolution" should "be good in gradient check for weight" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 4).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](layer, input, 1e-3) should be(true)
  }

  "A VolumetricConvolution with batch" should "be good in gradient check for weight" in {
    torchCheck()
    val seed = 100
    RNG.setSeed(seed)
    val from = RNG.uniform(2, 6).toInt
    val to = RNG.uniform(1, 4).toInt
    val kt = RNG.uniform(1, 4).toInt
    val ki = RNG.uniform(1, 4).toInt
    val kj = RNG.uniform(1, 4).toInt
    val st = RNG.uniform(1, 3).toInt
    val si = RNG.uniform(1, 3).toInt
    val sj = RNG.uniform(1, 3).toInt
    val padT = RNG.uniform(0, 2).toInt
    val padW = RNG.uniform(0, 2).toInt
    val padH = RNG.uniform(0, 2).toInt
    val outt = RNG.uniform(5, 7).toInt
    val outi = RNG.uniform(5, 7).toInt
    val outj = RNG.uniform(5, 7).toInt
    val batch = RNG.uniform(2, 7).toInt
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val layer = new VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
      padT, padW, padH)

    val input = Tensor[Double](batch, from, int, ini, inj).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](layer, input, 1e-3) should be(true)
  }

  "VolumetricConvolution L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val seed = 100
    RNG.setSeed(seed)
    val from = 3
    val to = 2
    val kt = 2
    val ki = 2
    val kj = 2
    val st = 2
    val si = 2
    val sj = 2
    val padT = 1
    val padW = 1
    val padH = 1
    val outt = 6
    val outi = 6
    val outj = 6
    val int = (outt - 1) * st + kt - padT * 2
    val ini = (outi - 1) * si + ki - padW * 2
    val inj = (outj - 1) * sj + kj - padH * 2
    val batch = 3

    val input = Tensor[Double](batch, from, int, inj, ini).apply1(e => Random.nextDouble())


    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val criterion = new MSECriterion[Double]

    val labels = Tensor[Double](1296).rand()

    val model1 = Sequential()
      .add(VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
        padT, padW, padH))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(VolumetricConvolution[Double](from, to, kt, ki, kj, st, si, sj,
        padT, padW, padH,
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

  "A VolumetricConvolution layer" should "work with SAME padding" in {
    import tensor.TensorNumericMath.TensorNumeric.NumericFloat
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 1
    val kT = 1
    val dT = 2
    val dW = 2
    val dH = 2
    val padW = -1
    val padH = -1
    val padT = -1
    val layer = new VolumetricConvolution(nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)

    val inputData = Array(
      0.0f, 1.0f, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
      20, 21, 22, 23, 24, 25, 26
    )

    val kernelData = Array(
      1.0f
    )

    val biasData = Array(0.0f)

    layer.weight.copy(Tensor(Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kT, kH, kW)))
    layer.bias.copy(Tensor(Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor(Storage(inputData), 1, Array(1, 3, 3, 3))
    val output = layer.updateOutput(input)
    val gradInput = layer.backward(input, output)
    output.storage().array() should be (Array(0.0f, 2, 6, 8, 18, 20, 24, 26))
  }

  "VolumetricConvolution computeOutputShape" should "work properly" in {
    val layer = VolumetricConvolution[Float](3, 8, 2, 1, 2)
    TestUtils.compareOutputShape(layer, Shape(3, 24, 28, 32)) should be (true)
  }
}

