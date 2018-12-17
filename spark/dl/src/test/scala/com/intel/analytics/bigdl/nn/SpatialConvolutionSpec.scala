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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

import scala.math._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random
import com.intel.analytics.bigdl.utils.{Shape, T, TestUtils}

@com.intel.analytics.bigdl.tags.Parallel
class SpatialConvolutionSpec extends FlatSpec with Matchers {
  "SpatialConvolution L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

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
      .add(new SpatialConvolution[Double](nInputPlane, nOutputPlane,
        kW, kH, dW, dH, padW, padH))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(new SpatialConvolution[Double](nInputPlane, nOutputPlane,
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


  "SpatialConvolution L2 regularizer set outside" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

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
      .add(new SpatialConvolution[Double](nInputPlane, nOutputPlane,
        kW, kH, dW, dH, padW, padH))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val conv = SpatialConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)
    conv.wRegularizer = L2Regularizer(0.1)
    conv.bRegularizer = L2Regularizer(0.1)
    val model2 = Sequential()
      .add(conv)
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

  "A SpatialConvolution layer" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }

  "A SpatialConvolution layer" should "work with SAME padding using NHWC format" in {
    import tensor.TensorNumericMath.TensorNumeric.NumericFloat
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = -1
    val padH = -1
    val layer = new SpatialConvolution(nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH, format = DataFormat.NHWC)

    val inputData = Array(
      1.0f, 2, 3, 4
    )

    val kernelData = Array(
      1.0f, 2, 3, 4
    )

    val biasData = Array(0.0f)

    layer.weight.copy(Tensor(Storage(kernelData), 1, Array(kH, kW, nOutputPlane,
      nInputPlane)))
    layer.bias.copy(Tensor(Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor(Storage(inputData), 1, Array(2, 2, 1))
    val output = layer.updateOutput(input)
    val gradInput = layer.backward(input, output)
    output.storage().array() should be (Array(30.0f, 14, 11, 4))
    gradInput.storage().array() should be (Array(30.0f, 74, 101, 188))
  }

  "A SpatialConvolution layer" should "work with SAME padding using NCHW format" in {
    import tensor.TensorNumericMath.TensorNumeric.NumericFloat
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 1
    val dW = 2
    val dH = 2
    val padW = -1
    val padH = -1
    val layer = new SpatialConvolution(nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      0.0f, 1.0f, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )

    val kernelData = Array(
      1.0f
    )

    val biasData = Array(0.0f)

    layer.weight.copy(Tensor(Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    layer.bias.copy(Tensor(Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor(Storage(inputData), 1, Array(1, 4, 4))
    val output = layer.updateOutput(input)
    val gradInput = layer.backward(input, output)
    output.storage().array() should be (Array(0.0f, 2, 8, 10))
    gradInput.storage().array() should be (Array(
      0.0f, 0, 2, 0, 0, 0, 0, 0, 8, 0, 10, 0, 0, 0, 0, 0
    ))
  }

  "A SpatialConvolution layer" should "generate same output with NCHW and NHWC" in {
    import tensor.TensorNumericMath.TensorNumeric.NumericDouble
    case class Conv(nIn: Int, nOut: Int, kW: Int,
                    kH: Int, dW: Int, dH: Int, pW: Int, pH: Int)
    val params = List(
      Conv(1, 1, 1, 1, 2, 2, -1, -1),
      Conv(1, 1, 3, 3, 1, 1, 0, 0),
      Conv(1, 1, 1, 1, 1, 1, 0, 0),
      Conv(1, 1, 5, 5, 1, 1, 0, 0),
      Conv(4, 4, 3, 3, 1, 1, 0, 0),
      Conv(4, 4, 1, 1, 1, 1, 0, 0),
      Conv(4, 4, 5, 5, 1, 1, 0, 0),
      Conv(4, 4, 3, 3, 2, 2, 0, 0),
      Conv(4, 4, 1, 1, 2, 2, 0, 0),
      Conv(4, 4, 5, 5, 2, 2, 0, 0),
      Conv(4, 4, 3, 3, 2, 2, 1, 1),
      Conv(4, 4, 1, 1, 2, 2, 1, 1),
      Conv(4, 4, 5, 5, 2, 2, 1, 1),
      Conv(4, 4, 1, 1, 2, 2, -1, -1),
      Conv(4, 4, 5, 5, 2, 2, -1, -1),
      Conv(1, 1, 2, 2, 1, 1, -1, -1)
    )

    for (param <- params) {
      println(param)
      val layer = new SpatialConvolution(param.nIn, param.nOut,
        param.kW, param.kH, param.dW, param.dH, param.pW, param.pH)
      val layerNHWC = new SpatialConvolution(param.nIn, param.nOut,
        param.kW, param.kH, param.dW, param.dH, param.pW, param.pH, format = DataFormat.NHWC)

      val input = Tensor(Array(4, param.nIn, 7, 7)).randn()

      val inputNHWC = Tensor(input.size())
        .copy(input).transpose(2, 4).transpose(2, 3).contiguous()

      val kernel = Tensor(Array(param.nOut, param.nIn, param.kH, param.kW)).randn()
      val bias = Tensor(Array(param.nOut)).randn()

      val kernelNHWC = Tensor(Array(1, param.nOut, param.nIn, param.kH, param.kW))
        .copy(kernel).transpose(2, 5).transpose(3, 4).transpose(2, 3).contiguous()
      val biasNHWC = Tensor(Array(param.nOut)).copy(bias)

      layer.weight.copy(kernel)
      layerNHWC.weight.copy(kernelNHWC)
      layer.bias.copy(bias)
      layerNHWC.bias.copy(biasNHWC)

      val output = layer.forward(input)
      val outputNHWC = layerNHWC.forward(inputNHWC)
      val gradOutput = Tensor(output.size()).fill(1.0)
      val gradOutputNHWC = Tensor(outputNHWC.size()).fill(1.0)

      val gradInput = layer.backward(input, gradOutput)
      val gradInputNHWC = layerNHWC.backward(inputNHWC, gradOutputNHWC)


      outputNHWC.transpose(2, 4).transpose(3, 4)
        .sub(output).pow(2).sum() should be < 1e-7

      gradInputNHWC.transpose(2, 4).transpose(3, 4)
        .sub(gradInput).pow(2).sum() should be < 1e-7

      val (weight1, grad1) = layer.getParameters()
      weight1.add(-0.01, grad1)
      val (weight2, grad2) = layerNHWC.getParameters()
      weight2.add(-0.01, grad2)

      val transWeight = layerNHWC.weight.transpose(2, 5).transpose(3, 4).transpose(4, 5)
      transWeight.sub(layer.weight).pow(2).sum() should be < 1e-7

    }
  }

  "A SpatialConvolution layer" should "generate correct output with given weight" in {
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

    val weight = Tensor[Double](T(
      T(2.0, 3.0),
      T(4.0, 5.0)
    ))

    val bias = Tensor[Double](T(0.0))

    val layer = new SpatialConvolution[Double](
      nInputPlane = nInputPlane,
      nOutputPlane = nOutputPlane,
      kernelW = kW,
      kernelH = kH,
      strideW = dW,
      strideH = dH,
      padW = padW,
      padH = padH,
      initWeight = weight,
      initBias = bias
    )

    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }

  it should "generate correct output when group != 1" in {
    val input1 = Tensor[Double](3, 4, 5).rand()
    val input2 = Tensor[Double](3, 4, 5).rand()

    val input = Tensor[Double](6, 4, 5).rand()
    input.narrow(1, 1, 3).copy(input1)
    input.narrow(1, 4, 3).copy(input2)

    val layer1 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer1.bias.fill(0)
    val output1 = layer1.updateOutput(input1)

    val layer2 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer2.bias.fill(0)
    val output2 = layer2.updateOutput(input2)

    val layer = new SpatialConvolution[Double](6, 8,
      2, 2, 1, 1, 0, 0, 2)
    layer.weight.select(1, 1).copy(layer1.weight.select(1, 1))
    layer.weight.select(1, 2).copy(layer2.weight.select(1, 1))
    layer.bias.fill(0)
    val output = layer.updateOutput(input)

    val targetOutput = Tensor[Double](output1.size(1) * 2, output1.size(2), output1.size(3))
    targetOutput.narrow(1, 1, 4).copy(output1)
    targetOutput.narrow(1, 5, 4).copy(output2)

    output should be(targetOutput)
  }

  it should "generate correct output when kernel is 1x1" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 1
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(2.0)
    output(Array(1, 1, 2)) should be(4.0)
    output(Array(1, 1, 3)) should be(6.0)
    output(Array(1, 2, 1)) should be(8.0)
    output(Array(1, 2, 2)) should be(10.0)
    output(Array(1, 2, 3)) should be(12.0)
    output(Array(1, 3, 1)) should be(14.0)
    output(Array(1, 3, 2)) should be(16.0)
    output(Array(1, 3, 3)) should be(18.0)
  }

  it should "generate correct output for batch input" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(3, 1, 3, 4))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1, 1)) should be(49)
    output(Array(1, 1, 1, 2)) should be(63)
    output(Array(1, 1, 1, 3)) should be(38)
    output(Array(1, 1, 2, 1)) should be(91)
    output(Array(1, 1, 2, 2)) should be(105)
    output(Array(1, 1, 2, 3)) should be(56)
    output(Array(2, 1, 1, 1)) should be(49)
    output(Array(2, 1, 1, 2)) should be(63)
    output(Array(2, 1, 1, 3)) should be(38)
    output(Array(2, 1, 2, 1)) should be(91)
    output(Array(2, 1, 2, 2)) should be(105)
    output(Array(2, 1, 2, 3)) should be(56)
    output(Array(3, 1, 1, 1)) should be(49)
    output(Array(3, 1, 1, 2)) should be(63)
    output(Array(3, 1, 1, 3)) should be(38)
    output(Array(3, 1, 2, 1)) should be(91)
    output(Array(3, 1, 2, 2)) should be(105)
    output(Array(3, 1, 2, 3)) should be(56)
  }

  it should "generate correct output for batch input when kernel size is 1" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 1
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1,
      1.0, 2, 3, 1,
      4, 5, 6, 1,
      7, 8, 9, 1
    )

    val kernelData = Array(
      2.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(3, 1, 3, 4))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1, 1)) should be(2)
    output(Array(1, 1, 1, 2)) should be(4)
    output(Array(1, 1, 1, 3)) should be(6)
    output(Array(1, 1, 1, 4)) should be(2)
    output(Array(1, 1, 2, 1)) should be(8)
    output(Array(1, 1, 2, 2)) should be(10)
    output(Array(1, 1, 2, 3)) should be(12)
    output(Array(1, 1, 2, 4)) should be(2)
    output(Array(1, 1, 3, 1)) should be(14)
    output(Array(1, 1, 3, 2)) should be(16)
    output(Array(1, 1, 3, 3)) should be(18)
    output(Array(1, 1, 3, 4)) should be(2)
    output(Array(2, 1, 1, 1)) should be(2)
    output(Array(2, 1, 1, 2)) should be(4)
    output(Array(2, 1, 1, 3)) should be(6)
    output(Array(2, 1, 1, 4)) should be(2)
    output(Array(2, 1, 2, 1)) should be(8)
    output(Array(2, 1, 2, 2)) should be(10)
    output(Array(2, 1, 2, 3)) should be(12)
    output(Array(2, 1, 2, 4)) should be(2)
    output(Array(2, 1, 3, 1)) should be(14)
    output(Array(2, 1, 3, 2)) should be(16)
    output(Array(2, 1, 3, 3)) should be(18)
    output(Array(2, 1, 3, 4)) should be(2)
    output(Array(3, 1, 1, 1)) should be(2)
    output(Array(3, 1, 1, 2)) should be(4)
    output(Array(3, 1, 1, 3)) should be(6)
    output(Array(3, 1, 1, 4)) should be(2)
    output(Array(3, 1, 2, 1)) should be(8)
    output(Array(3, 1, 2, 2)) should be(10)
    output(Array(3, 1, 2, 3)) should be(12)
    output(Array(3, 1, 2, 4)) should be(2)
    output(Array(3, 1, 3, 1)) should be(14)
    output(Array(3, 1, 3, 2)) should be(16)
    output(Array(3, 1, 3, 3)) should be(18)
    output(Array(3, 1, 3, 4)) should be(2)
  }

  it should "generate correct output when group != 1 for batch input" in {
    val input1 = Tensor[Double](4, 3, 4, 5).rand()
    val input2 = Tensor[Double](4, 3, 4, 5).rand()

    val input = Tensor[Double](4, 6, 4, 5)
    input.narrow(2, 1, 3).copy(input1)
    input.narrow(2, 4, 3).copy(input2)

    val layer1 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer1.bias.fill(0)
    val output1 = layer1.updateOutput(input1)

    val layer2 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer2.bias.fill(0)
    val output2 = layer2.updateOutput(input2)

    val layer = new SpatialConvolution[Double](6, 8,
      2, 2, 1, 1, 0, 0, 2)
    layer.weight.select(1, 1).copy(layer1.weight.select(1, 1))
    layer.weight.select(1, 2).copy(layer2.weight.select(1, 1))
    layer.bias.fill(0)
    val output = layer.updateOutput(input)

    val targetOutput = Tensor[Double](output1.size(1), output1.size(2) * 2, output1.size(3),
      output1.size(4))
    targetOutput.narrow(2, 1, 4).copy(output1)
    targetOutput.narrow(2, 5, 4).copy(output2)

    output should be(targetOutput)
  }

  it should "generate correct output when there's offset" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      0.0, 0.0, 1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 3, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }

  it should "generate correct output even called twice" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }

  it should "generate correct output when there's bias" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(1.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(50)
    output(Array(1, 1, 2)) should be(64)
    output(Array(1, 2, 1)) should be(92)
    output(Array(1, 2, 2)) should be(106)
  }

  "A SpatialConvolution with strideW" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 2
    val dW = 2
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0,
      4
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(18)
    output(Array(1, 1, 2)) should be(30)
    output(Array(1, 2, 1)) should be(36)
    output(Array(1, 2, 2)) should be(48)
  }

  "A SpatialConvolution with strideH" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 1
    val dW = 1
    val dH = 2
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(8)
    output(Array(1, 1, 2)) should be(13)
    output(Array(1, 2, 1)) should be(38)
    output(Array(1, 2, 2)) should be(43)
  }

  "A SpatialConvolution with padW" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 1
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(23)
    output(Array(1, 1, 2)) should be(49)
    output(Array(1, 1, 3)) should be(63)
    output(Array(1, 1, 4)) should be(30)
    output(Array(1, 2, 1)) should be(47)
    output(Array(1, 2, 2)) should be(91)
    output(Array(1, 2, 3)) should be(105)
    output(Array(1, 2, 4)) should be(48)
  }

  "A SpatialConvolution with padH" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 1
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(14)
    output(Array(1, 1, 2)) should be(23)
    output(Array(1, 2, 1)) should be(49)
    output(Array(1, 2, 2)) should be(63)
    output(Array(1, 3, 1)) should be(91)
    output(Array(1, 3, 2)) should be(105)
    output(Array(1, 4, 1)) should be(38)
    output(Array(1, 4, 2)) should be(43)
  }

  "A SpatialConvolution with multiple plane" should "generate correct output" in {
    val nInputPlane = 2
    val nOutputPlane = 2
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0, 0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49 * 2)
    output(Array(1, 1, 2)) should be(63 * 2)
    output(Array(1, 2, 1)) should be(91 * 2)
    output(Array(1, 2, 2)) should be(105 * 2)
    output(Array(2, 1, 1)) should be(49 * 2)
    output(Array(2, 1, 2)) should be(63 * 2)
    output(Array(2, 2, 1)) should be(91 * 2)
    output(Array(2, 2, 2)) should be(105 * 2)
  }

  "A SpatialConvolution with multiple plane with different bias" should "get correct output" in {
    val nInputPlane = 2
    val nOutputPlane = 2
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0, 1.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49 * 2)
    output(Array(1, 1, 2)) should be(63 * 2)
    output(Array(1, 2, 1)) should be(91 * 2)
    output(Array(1, 2, 2)) should be(105 * 2)
    output(Array(2, 1, 1)) should be(49 + 50)
    output(Array(2, 1, 2)) should be(63 + 64)
    output(Array(2, 2, 1)) should be(91 + 92)
    output(Array(2, 2, 2)) should be(105 + 106)
  }

  "A SpatialConvolution with different input/output plane" should "generate correct output" in {
    val nInputPlane = 2
    val nOutputPlane = 3
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0, 0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49 * 2)
    output(Array(1, 1, 2)) should be(63 * 2)
    output(Array(1, 2, 1)) should be(91 * 2)
    output(Array(1, 2, 2)) should be(105 * 2)
    output(Array(2, 1, 1)) should be(49 * 2)
    output(Array(2, 1, 2)) should be(63 * 2)
    output(Array(2, 2, 1)) should be(91 * 2)
    output(Array(2, 2, 2)) should be(105 * 2)
    output(Array(3, 1, 1)) should be(49 * 2)
    output(Array(3, 1, 2)) should be(63 * 2)
    output(Array(3, 2, 1)) should be(91 * 2)
    output(Array(3, 2, 2)) should be(105 * 2)
  }

  "A SpatialConvolution" should "generate correct gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 2)) should be(7)
    gradInput(Array(1, 1, 3)) should be(6)
    gradInput(Array(1, 2, 1)) should be(10)
    gradInput(Array(1, 2, 2)) should be(30)
    gradInput(Array(1, 2, 3)) should be(22)
    gradInput(Array(1, 3, 1)) should be(12)
    gradInput(Array(1, 3, 2)) should be(31)
    gradInput(Array(1, 3, 3)) should be(20)
  }

  it should "generate correct gradInput when kernel size is 1x1" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 1
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0
    )

    val gradOutputData = Array(
      1.0, 2.0, 5.0,
      3.0, 4.0, 6.0,
      7.0, 8.0, 9.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 3, 3))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 2)) should be(4)
    gradInput(Array(1, 1, 3)) should be(10)
    gradInput(Array(1, 2, 1)) should be(6)
    gradInput(Array(1, 2, 2)) should be(8)
    gradInput(Array(1, 2, 3)) should be(12)
    gradInput(Array(1, 3, 1)) should be(14)
    gradInput(Array(1, 3, 2)) should be(16)
    gradInput(Array(1, 3, 3)) should be(18)
  }

  it should "generate correct gradInput when group != 1" in {
    val input1 = Tensor[Double](3, 4, 5).rand()
    val gradOutput1 = Tensor[Double](4, 3, 4).rand()
    val input2 = Tensor[Double](3, 4, 5).rand()
    val gradOutput2 = Tensor[Double](4, 3, 4).rand()

    val input = Tensor[Double](6, 4, 5).rand()
    input.narrow(1, 1, 3).copy(input1)
    input.narrow(1, 4, 3).copy(input2)
    val gradOutput = Tensor[Double](8, 3, 4).rand()
    gradOutput.narrow(1, 1, 4).copy(gradOutput1)
    gradOutput.narrow(1, 5, 4).copy(gradOutput2)

    val layer1 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer1.bias.fill(0)
    layer1.updateOutput(input1)
    val gradInput1 = layer1.updateGradInput(input1, gradOutput1)

    val layer2 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer2.bias.fill(0)
    layer2.updateOutput(input2)
    val gradInput2 = layer2.updateGradInput(input2, gradOutput2)


    val layer = new SpatialConvolution[Double](6, 8,
      2, 2, 1, 1, 0, 0, 2)
    layer.weight.select(1, 1).copy(layer1.weight.select(1, 1))
    layer.weight.select(1, 2).copy(layer2.weight.select(1, 1))
    layer.bias.fill(0)
    layer.updateOutput(input)
    val gradInput = layer.updateGradInput(input, gradOutput)

    val targetGradInput = Tensor[Double](gradInput1.size(1) * 2, gradInput1.size(2),
      gradInput1.size(3))
    targetGradInput.narrow(1, 1, 3).copy(gradInput1)
    targetGradInput.narrow(1, 4, 3).copy(gradInput2)

    gradInput should be(targetGradInput)
  }

  it should "generate correct gradInput for batch input" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0,
      1.0, 2.0,
      3.0, 4.0,
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(3, 1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(3, 1, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 1, 2)) should be(7)
    gradInput(Array(1, 1, 1, 3)) should be(6)
    gradInput(Array(1, 1, 2, 1)) should be(10)
    gradInput(Array(1, 1, 2, 2)) should be(30)
    gradInput(Array(1, 1, 2, 3)) should be(22)
    gradInput(Array(1, 1, 3, 1)) should be(12)
    gradInput(Array(1, 1, 3, 2)) should be(31)
    gradInput(Array(1, 1, 3, 3)) should be(20)
    gradInput(Array(2, 1, 1, 1)) should be(2)
    gradInput(Array(2, 1, 1, 2)) should be(7)
    gradInput(Array(2, 1, 1, 3)) should be(6)
    gradInput(Array(2, 1, 2, 1)) should be(10)
    gradInput(Array(2, 1, 2, 2)) should be(30)
    gradInput(Array(2, 1, 2, 3)) should be(22)
    gradInput(Array(2, 1, 3, 1)) should be(12)
    gradInput(Array(2, 1, 3, 2)) should be(31)
    gradInput(Array(2, 1, 3, 3)) should be(20)
    gradInput(Array(3, 1, 1, 1)) should be(2)
    gradInput(Array(3, 1, 1, 2)) should be(7)
    gradInput(Array(3, 1, 1, 3)) should be(6)
    gradInput(Array(3, 1, 2, 1)) should be(10)
    gradInput(Array(3, 1, 2, 2)) should be(30)
    gradInput(Array(3, 1, 2, 3)) should be(22)
    gradInput(Array(3, 1, 3, 1)) should be(12)
    gradInput(Array(3, 1, 3, 2)) should be(31)
    gradInput(Array(3, 1, 3, 3)) should be(20)
  }

  it should "generate correct gradInput for batch input when kernel is 1x1" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 1
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0
    )

    val gradOutputData = Array(
      1.0, 2.0, 4.0,
      3.0, 4.0, 7.0,
      8.0, 6.0, 9.0,
      1.0, 2.0, 4.0,
      3.0, 4.0, 7.0,
      8.0, 6.0, 9.0,
      1.0, 2.0, 4.0,
      3.0, 4.0, 7.0,
      8.0, 6.0, 9.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(3, 1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(3, 1, 3, 3))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 1, 2)) should be(4)
    gradInput(Array(1, 1, 1, 3)) should be(8)
    gradInput(Array(1, 1, 2, 1)) should be(6)
    gradInput(Array(1, 1, 2, 2)) should be(8)
    gradInput(Array(1, 1, 2, 3)) should be(14)
    gradInput(Array(1, 1, 3, 1)) should be(16)
    gradInput(Array(1, 1, 3, 2)) should be(12)
    gradInput(Array(1, 1, 3, 3)) should be(18)
    gradInput(Array(2, 1, 1, 1)) should be(2)
    gradInput(Array(2, 1, 1, 2)) should be(4)
    gradInput(Array(2, 1, 1, 3)) should be(8)
    gradInput(Array(2, 1, 2, 1)) should be(6)
    gradInput(Array(2, 1, 2, 2)) should be(8)
    gradInput(Array(2, 1, 2, 3)) should be(14)
    gradInput(Array(2, 1, 3, 1)) should be(16)
    gradInput(Array(2, 1, 3, 2)) should be(12)
    gradInput(Array(2, 1, 3, 3)) should be(18)
    gradInput(Array(3, 1, 1, 1)) should be(2)
    gradInput(Array(3, 1, 1, 2)) should be(4)
    gradInput(Array(3, 1, 1, 3)) should be(8)
    gradInput(Array(3, 1, 2, 1)) should be(6)
    gradInput(Array(3, 1, 2, 2)) should be(8)
    gradInput(Array(3, 1, 2, 3)) should be(14)
    gradInput(Array(3, 1, 3, 1)) should be(16)
    gradInput(Array(3, 1, 3, 2)) should be(12)
    gradInput(Array(3, 1, 3, 3)) should be(18)
  }

  it should "generate correct gradInput when group != 1 for batch input" in {
    val input1 = Tensor[Double](4, 3, 4, 5).rand()
    val gradOutput1 = Tensor[Double](4, 4, 3, 4).rand()
    val input2 = Tensor[Double](4, 3, 4, 5).rand()
    val gradOutput2 = Tensor[Double](4, 4, 3, 4).rand()

    val input = Tensor[Double](4, 6, 4, 5).rand()
    input.narrow(2, 1, 3).copy(input1)
    input.narrow(2, 4, 3).copy(input2)
    val gradOutput = Tensor[Double](4, 8, 3, 4).rand()
    gradOutput.narrow(2, 1, 4).copy(gradOutput1)
    gradOutput.narrow(2, 5, 4).copy(gradOutput2)

    val layer1 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer1.bias.fill(0)
    layer1.updateOutput(input1)
    val gradInput1 = layer1.updateGradInput(input1, gradOutput1)

    val layer2 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer2.bias.fill(0)
    layer2.updateOutput(input2)
    val gradInput2 = layer2.updateGradInput(input2, gradOutput2)


    val layer = new SpatialConvolution[Double](6, 8,
      2, 2, 1, 1, 0, 0, 2)
    layer.weight.select(1, 1).copy(layer1.weight.select(1, 1))
    layer.weight.select(1, 2).copy(layer2.weight.select(1, 1))
    layer.bias.fill(0)
    layer.updateOutput(input)
    val gradInput = layer.updateGradInput(input, gradOutput)

    val targetGradInput = Tensor[Double](gradInput1.size(1), gradInput1.size(2) * 2,
      gradInput1.size(3), gradInput1.size(4))
    targetGradInput.narrow(2, 1, 3).copy(gradInput1)
    targetGradInput.narrow(2, 4, 3).copy(gradInput2)

    gradInput should be(targetGradInput)
  }

  "A SpatialConvolution with offset" should "generate correct gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      0.0, 0.0, 0.0, 1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      0.0, 2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      0.0, 0.0, 1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 2,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 4, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 3, Array(1, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 2)) should be(7)
    gradInput(Array(1, 1, 3)) should be(6)
    gradInput(Array(1, 2, 1)) should be(10)
    gradInput(Array(1, 2, 2)) should be(30)
    gradInput(Array(1, 2, 3)) should be(22)
    gradInput(Array(1, 3, 1)) should be(12)
    gradInput(Array(1, 3, 2)) should be(31)
    gradInput(Array(1, 3, 3)) should be(20)
  }

  "A SpatialConvolution" should "generate correct gradInput even called twice" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.updateOutput(input)
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 2)) should be(7)
    gradInput(Array(1, 1, 3)) should be(6)
    gradInput(Array(1, 2, 1)) should be(10)
    gradInput(Array(1, 2, 2)) should be(30)
    gradInput(Array(1, 2, 3)) should be(22)
    gradInput(Array(1, 3, 1)) should be(12)
    gradInput(Array(1, 3, 2)) should be(31)
    gradInput(Array(1, 3, 3)) should be(20)
  }

  "A SpatialConvolution with strideW" should "generate correct gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 1
    val kH = 2
    val dW = 2
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0,
      4
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 2)) should be(0)
    gradInput(Array(1, 1, 3)) should be(4)
    gradInput(Array(1, 2, 1)) should be(10)
    gradInput(Array(1, 2, 2)) should be(0)
    gradInput(Array(1, 2, 3)) should be(16)
    gradInput(Array(1, 3, 1)) should be(12)
    gradInput(Array(1, 3, 2)) should be(0)
    gradInput(Array(1, 3, 3)) should be(16)
  }

  "A SpatialConvolution with strideH" should "generate correct gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 1
    val dW = 1
    val dH = 2
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(2)
    gradInput(Array(1, 1, 2)) should be(7)
    gradInput(Array(1, 1, 3)) should be(6)
    gradInput(Array(1, 2, 1)) should be(0)
    gradInput(Array(1, 2, 2)) should be(0)
    gradInput(Array(1, 2, 3)) should be(0)
    gradInput(Array(1, 3, 1)) should be(6)
    gradInput(Array(1, 3, 2)) should be(17)
    gradInput(Array(1, 3, 3)) should be(12)
  }

  "A SpatialConvolution with padW" should "generate correct gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 1
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0, 1, 2,
      3.0, 4.0, 3, 4
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 4))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(7)
    gradInput(Array(1, 1, 2)) should be(8)
    gradInput(Array(1, 1, 3)) should be(7)
    gradInput(Array(1, 2, 1)) should be(30)
    gradInput(Array(1, 2, 2)) should be(32)
    gradInput(Array(1, 2, 3)) should be(30)
    gradInput(Array(1, 3, 1)) should be(31)
    gradInput(Array(1, 3, 2)) should be(32)
    gradInput(Array(1, 3, 3)) should be(31)
  }

  "A SpatialConvolution with padH" should "generate correct gradInput" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 1
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1, 2,
      1.0, 2.0,
      3.0, 4.0,
      1, 2
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 4, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(6)
    gradInput(Array(1, 1, 2)) should be(20)
    gradInput(Array(1, 1, 3)) should be(16)
    gradInput(Array(1, 2, 1)) should be(10)
    gradInput(Array(1, 2, 2)) should be(30)
    gradInput(Array(1, 2, 3)) should be(22)
    gradInput(Array(1, 3, 1)) should be(14)
    gradInput(Array(1, 3, 2)) should be(38)
    gradInput(Array(1, 3, 3)) should be(26)
  }

  "A SpatialConvolution with multiple plane" should "generate correct gradInput" in {
    val nInputPlane = 2
    val nOutputPlane = 2
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )


    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0,

      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0, 0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(2, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(4)
    gradInput(Array(1, 1, 2)) should be(14)
    gradInput(Array(1, 1, 3)) should be(12)
    gradInput(Array(1, 2, 1)) should be(20)
    gradInput(Array(1, 2, 2)) should be(60)
    gradInput(Array(1, 2, 3)) should be(44)
    gradInput(Array(1, 3, 1)) should be(24)
    gradInput(Array(1, 3, 2)) should be(62)
    gradInput(Array(1, 3, 3)) should be(40)
    gradInput(Array(2, 1, 1)) should be(4)
    gradInput(Array(2, 1, 2)) should be(14)
    gradInput(Array(2, 1, 3)) should be(12)
    gradInput(Array(2, 2, 1)) should be(20)
    gradInput(Array(2, 2, 2)) should be(60)
    gradInput(Array(2, 2, 3)) should be(44)
    gradInput(Array(2, 3, 1)) should be(24)
    gradInput(Array(2, 3, 2)) should be(62)
    gradInput(Array(2, 3, 3)) should be(40)
  }

  "A SpatialConvolution with different input/output plane and input width/height" should
    "generate correct gradInput" in {
    val nInputPlane = 2
    val nOutputPlane = 3
    val kW = 3
    val kH = 2
    val dW = 2
    val dH = 3
    val padW = 1
    val padH = 2
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      0.31065078778192, 0.47680206806399, 0.47333685657941, 0.60491567384452, 0.45523861795664,
      0.59684172924608,
      0.18907571188174, 0.86919780098833, 0.10920453490689, 0.32645826041698, 0.34624883672222,
      0.82086082990281,
      0.2254743387457, 0.054266506806016, 0.44735967810266, 0.18092241417617, 0.77504938514903,
      0.84614481125027,
      0.38507605646737, 0.01930161495693, 0.71951506263576, 0.51190832140855, 0.29157469817437,
      0.96515286993235,
      0.55696543189697, 0.39736612164415, 0.37251793802716, 0.94657975435257, 0.78204862540588,
      0.81618627184071,

      0.21099418401718, 0.94533503637649, 0.18719119438902, 0.439231029246, 0.6292649700772,
      0.49809367349371,
      0.84911825321615, 0.66236829245463, 0.22733291704208, 0.20995571650565, 0.20888273068704,
      0.022939298069105,
      0.026140305912122, 0.78018275159411, 0.34802454058081, 0.10564375855029, 0.68778858194128,
      0.19946397026069,
      0.22409740672447, 0.93569488637149, 0.12785892537795, 0.15872345422395, 0.59272385016084,
      0.29062455683015,
      0.41214634198695, 0.37712478893809, 0.96965333609842, 0.76970787090249, 0.4584473001305,
      0.55620927736163
    )

    val kernelData = Array(
      0.076179399003951, 0.011624260825773, -0.26929373771844,
      0.21727415347489, -0.12130985860073, -0.075744581174401,

      0.13516901658924, -0.11696029209996, -0.22248883304562,
      0.023747188857974, -0.21052245873008, -0.076731029460861,

      0.27798097331555, -0.21490849912455, 0.15393807971066,
      -0.10031578297795, -0.28516644100567, -0.19091797389294,

      0.12532535545942, -0.072735155267421, -0.013593492171148,
      0.066118103528912, -0.12672802914211, 0.19073741499255,

      -0.28819224469682, -0.18149741183885, 0.036125793463643,
      0.16313411236647, 0.24139356738889, 0.12428446592142,

      0.00075597771376307, 0.24760437974888, 0.017539558871527,
      0.0097350586336657, 0.16691246244305, 0.089457577292526
    )


    val gradOutputData = Array(
      0.62875230400823, 0.32400949206203, 0.34562376490794,
      0.48625181033276, 0.99999123509042, 0.13880217191763,
      0.536536491476, 0.79385008965619, 0.88393420679495,

      0.16750993672758, 0.87882893625647, 0.43605240457691,
      0.6109996503219, 0.18711599125527, 0.39965781872161,
      0.0030532660894096, 0.44951300788671, 0.69211066956632,

      0.68326092790812, 0.13290155655704, 0.019294884288684,
      0.27066777111031, 0.56331930914894, 0.040327817667276,
      0.53744582436047, 0.54968754481524, 0.045935253845528
    )

    val biasData = Array(-0.2619934213478, 0.28232013358659, -0.048443703240033)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 5, 6))
    val output = layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1,
      Array(nOutputPlane, 3, 3))
    val gradInput = layer.updateGradInput(input, gradOutput)
    val expectedGradData = Array(
      0, 0, 0, 0, 0, 0,
      -0.17478219987071, -0.061261206120279, -0.13082965455208, -0.11008777280852,
      -0.091595783867018, 0.025600875888105,
      -0.1678862752483, 0.1705562017599, -0.03868633899779, -0.044810903598228,
      -0.12107219386202, -0.081803252098247,
      0, 0, 0, 0, 0, 0,
      -0.091964358838267, -0.097584847687116, -0.18714311206719, 0.12176920040528,
      -0.1468025131371, -0.12983631153158,

      0, 0, 0, 0, 0, 0,
      -0.034294782621376, 0.047300243300225, 0.008911150511847, -0.14627057232367,
      -0.035318171790018, -0.035607346551946,
      -0.13461988398503, 0.14504585663619, -0.14020844128075, 0.039466216288395,
      -0.073137606855201, 0.069186894547888,
      0, 0, 0, 0, 0, 0,
      0.070098395440994, 0.054066545221617, 0.010560706796531, 0.033162304609837,
      -0.14235221, -0.20526800703813
    )

    val expectedGrad = Tensor[Double](Storage(expectedGradData), 1, Array(2, 5, 6))
    expectedGrad.map(gradInput, (v1, v2) => {
      v1 should be(v2 +- 1e-6);
      v1
    })
  }

  "A SpatialConvolution with different input/output plane" should "generate correct gradInput" in {
    val nInputPlane = 2
    val nOutputPlane = 3
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )


    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0,

      1.0, 2.0,
      3.0, 4.0,

      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0, 0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1,
      Array(nOutputPlane, 2, 2))
    val gradInput = layer.updateGradInput(input, gradOutput)
    gradInput(Array(1, 1, 1)) should be(6)
    gradInput(Array(1, 1, 2)) should be(21)
    gradInput(Array(1, 1, 3)) should be(18)
    gradInput(Array(1, 2, 1)) should be(30)
    gradInput(Array(1, 2, 2)) should be(90)
    gradInput(Array(1, 2, 3)) should be(66)
    gradInput(Array(1, 3, 1)) should be(36)
    gradInput(Array(1, 3, 2)) should be(93)
    gradInput(Array(1, 3, 3)) should be(60)
    gradInput(Array(2, 1, 1)) should be(6)
    gradInput(Array(2, 1, 2)) should be(21)
    gradInput(Array(2, 1, 3)) should be(18)
    gradInput(Array(2, 2, 1)) should be(30)
    gradInput(Array(2, 2, 2)) should be(90)
    gradInput(Array(2, 2, 3)) should be(66)
    gradInput(Array(2, 3, 1)) should be(36)
    gradInput(Array(2, 3, 2)) should be(93)
    gradInput(Array(2, 3, 3)) should be(60)
  }

  "A SpatialConvolution" should "generate correct gradWeight" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 1, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 1, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 1, 1, 2, 2)) should be(77)

    layer.gradBias(Array(1)) should be(10)
  }

  it should "generate correct gradWeight when group != 1" in {
    val input1 = Tensor[Double](3, 4, 5).rand()
    val gradOutput1 = Tensor[Double](4, 3, 4).rand()
    val input2 = Tensor[Double](3, 4, 5).rand()
    val gradOutput2 = Tensor[Double](4, 3, 4).rand()

    val input = Tensor[Double](6, 4, 5).rand()
    input.narrow(1, 1, 3).copy(input1)
    input.narrow(1, 4, 3).copy(input2)
    val gradOutput = Tensor[Double](8, 3, 4).rand()
    gradOutput.narrow(1, 1, 4).copy(gradOutput1)
    gradOutput.narrow(1, 5, 4).copy(gradOutput2)

    val layer1 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer1.bias.fill(0)
    layer1.updateOutput(input1)
    val gradInput1 = layer1.updateGradInput(input1, gradOutput1)
    layer1.accGradParameters(input1, gradOutput1)

    val layer2 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer2.bias.fill(0)
    layer2.updateOutput(input2)
    val gradInput2 = layer2.updateGradInput(input2, gradOutput2)
    layer2.accGradParameters(input2, gradOutput2)


    val layer = new SpatialConvolution[Double](6, 8,
      2, 2, 1, 1, 0, 0, 2)
    layer.weight.select(1, 1).copy(layer1.weight.select(1, 1))
    layer.weight.select(1, 2).copy(layer2.weight.select(1, 1))
    layer.bias.fill(0)
    layer.updateOutput(input)
    val gradInput = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.weight.select(1, 1) should be(layer1.weight.select(1, 1))
    layer.weight.select(1, 2) should be(layer2.weight.select(1, 1))
  }

  it should "generate correct gradWeight for batch input" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9

    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0,
      1.0, 2.0,
      3.0, 4.0,
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(3, 1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(3, 1, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1, 1, 1, 1)) should be(37 * 3)
    layer.gradWeight(Array(1, 1, 1, 1, 2)) should be(47 * 3)
    layer.gradWeight(Array(1, 1, 1, 2, 1)) should be(67 * 3)
    layer.gradWeight(Array(1, 1, 1, 2, 2)) should be(77 * 3)

    layer.gradBias(Array(1)) should be(10 * 3)
  }

  it should "generate correct gradWeight when group != 1 for batch input" in {
    val input1 = Tensor[Double](4, 3, 4, 5).rand()
    val gradOutput1 = Tensor[Double](4, 4, 3, 4).rand()
    val input2 = Tensor[Double](4, 3, 4, 5).rand()
    val gradOutput2 = Tensor[Double](4, 4, 3, 4).rand()

    val input = Tensor[Double](4, 6, 4, 5).rand()
    input.narrow(2, 1, 3).copy(input1)
    input.narrow(2, 4, 3).copy(input2)
    val gradOutput = Tensor[Double](4, 8, 3, 4).rand()
    gradOutput.narrow(2, 1, 4).copy(gradOutput1)
    gradOutput.narrow(2, 5, 4).copy(gradOutput2)

    val layer1 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer1.bias.fill(0)
    layer1.updateOutput(input1)
    val gradInput1 = layer1.updateGradInput(input1, gradOutput1)
    layer1.accGradParameters(input1, gradOutput1)

    val layer2 = new SpatialConvolution[Double](3, 4,
      2, 2, 1, 1, 0, 0)
    layer2.bias.fill(0)
    layer2.updateOutput(input2)
    val gradInput2 = layer2.updateGradInput(input2, gradOutput2)
    layer2.accGradParameters(input2, gradOutput2)


    val layer = new SpatialConvolution[Double](6, 8,
      2, 2, 1, 1, 0, 0, 2)
    layer.weight.select(1, 1).copy(layer1.weight.select(1, 1))
    layer.weight.select(1, 2).copy(layer2.weight.select(1, 1))
    layer.bias.fill(0)
    layer.updateOutput(input)
    val gradInput = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.weight.select(1, 1) should be(layer1.weight.select(1, 1))
    layer.weight.select(1, 2) should be(layer2.weight.select(1, 1))
  }

  "A SpatialConvolution" should "generate correct gradWeight even called twice" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(1, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 1, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 1, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 1, 1, 2, 2)) should be(77)

    layer.gradBias(Array(1)) should be(10)
  }

  "A SpatialConvolution with multiple plane" should "generate correct gradWeight" in {
    val nInputPlane = 2
    val nOutputPlane = 2
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )


    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0,

      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0, 0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(2, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 1, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 1, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 1, 1, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 1, 2, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 1, 2, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 1, 2, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 1, 2, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 2, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 2, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 2, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 2, 1, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 2, 2, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 2, 2, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 2, 2, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 2, 2, 2, 2)) should be(77)
  }

  "A SpatialConvolution with different input/output plane" should "get correct gradWeight" in {
    val nInputPlane = 2
    val nOutputPlane = 3
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5,

      2.0, 3,
      4, 5
    )


    val gradOutputData = Array(
      1.0, 2.0,
      3.0, 4.0,

      1.0, 2.0,
      3.0, 4.0,

      1.0, 2.0,
      3.0, 4.0
    )

    val biasData = Array(0.0, 0.0, 0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(nOutputPlane, nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 3, 3))
    val output = layer.updateOutput(input)
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1,
      Array(nOutputPlane, 2, 2))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 1, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 1, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 1, 1, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 1, 2, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 1, 2, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 1, 2, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 1, 2, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 2, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 2, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 2, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 2, 1, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 2, 2, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 2, 2, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 2, 2, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 2, 2, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 3, 1, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 3, 1, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 3, 1, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 3, 1, 2, 2)) should be(77)
    layer.gradWeight(Array(1, 3, 2, 1, 1)) should be(37)
    layer.gradWeight(Array(1, 3, 2, 1, 2)) should be(47)
    layer.gradWeight(Array(1, 3, 2, 2, 1)) should be(67)
    layer.gradWeight(Array(1, 3, 2, 2, 2)) should be(77)
  }

  "A SpatialConvolution" should "generate correct gradWeight gradBias output gradInput" in {
    val weightData = Array(
      0.076179399003951, 0.011624260825773, -0.26929373771844,
      0.21727415347489, -0.12130985860073, -0.075744581174401,

      0.13516901658924, -0.11696029209996, -0.22248883304562,
      0.023747188857974, -0.21052245873008, -0.076731029460861,

      0.27798097331555, -0.21490849912455, 0.15393807971066,
      -0.10031578297795, -0.28516644100567, -0.19091797389294,

      0.12532535545942, -0.072735155267421, -0.013593492171148,
      0.066118103528912, -0.12672802914211, 0.19073741499255,

      -0.28819224469682, -0.18149741183885, 0.036125793463643,
      0.16313411236647, 0.24139356738889, 0.12428446592142,

      0.00075597771376307, 0.24760437974888, 0.017539558871527,
      0.0097350586336657, 0.16691246244305, 0.089457577292526
    )
    val weight = Tensor[Double](Storage(weightData), 1, Array(3, 2, 2, 3))
    val biasData = Array(
      -0.2619934213478, 0.28232013358659, -0.048443703240033
    )
    val bias = Tensor[Double](Storage(biasData), 1, Array(3))
    val module = new SpatialConvolution[Double](2, 3, 3, 2, 2, 3, 1, 2)
    module.weight.copy(weight)
    module.bias.copy(bias)
    val inputData = Array(
      0.31065078778192, 0.47680206806399, 0.47333685657941, 0.60491567384452, 0.45523861795664,
      0.59684172924608,
      0.18907571188174, 0.86919780098833, 0.10920453490689, 0.32645826041698, 0.34624883672222,
      0.82086082990281,
      0.2254743387457, 0.054266506806016, 0.44735967810266, 0.18092241417617, 0.77504938514903,
      0.84614481125027,
      0.38507605646737, 0.01930161495693, 0.71951506263576, 0.51190832140855, 0.29157469817437,
      0.96515286993235,
      0.55696543189697, 0.39736612164415, 0.37251793802716, 0.94657975435257, 0.78204862540588,
      0.81618627184071,

      0.21099418401718, 0.94533503637649, 0.18719119438902, 0.439231029246, 0.6292649700772,
      0.49809367349371,
      0.84911825321615, 0.66236829245463, 0.22733291704208, 0.20995571650565, 0.20888273068704,
      0.022939298069105,
      0.026140305912122, 0.78018275159411, 0.34802454058081, 0.10564375855029, 0.68778858194128,
      0.19946397026069,
      0.22409740672447, 0.93569488637149, 0.12785892537795, 0.15872345422395, 0.59272385016084,
      0.29062455683015,
      0.41214634198695, 0.37712478893809, 0.96965333609842, 0.76970787090249, 0.4584473001305,
      0.55620927736163
    )
    val input = Tensor[Double](Storage(inputData), 1, Array(2, 5, 6))
    val gradOutputData = Array(
      0.62875230400823, 0.32400949206203, 0.34562376490794,
      0.48625181033276, 0.99999123509042, 0.13880217191763,
      0.536536491476, 0.79385008965619, 0.88393420679495,

      0.16750993672758, 0.87882893625647, 0.43605240457691,
      0.6109996503219, 0.18711599125527, 0.39965781872161,
      0.0030532660894096, 0.44951300788671, 0.69211066956632,

      0.68326092790812, 0.13290155655704, 0.019294884288684,
      0.27066777111031, 0.56331930914894, 0.040327817667276,
      0.53744582436047, 0.54968754481524, 0.045935253845528
    )
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(3, 3, 3))
    val expectedOutputData = Array(
      -0.2619934213478, -0.2619934213478, -0.2619934213478,
      -0.83737774911124, -0.38522056607752, -0.73170056476513,
      -0.49463812878686, -0.71598681999743, -0.47391648638437,

      0.28232013358659, 0.28232013358659, 0.28232013358659,
      0.37556331587562, 0.47442123673519, -0.0070165134165954,
      0.18868933521803, 0.42471014791584, 0.55858239657379,

      -0.048443703240033, -0.048443703240033, -0.048443703240033,
      0.30583199116269, -0.032030947882614, 0.33200759191869,
      -0.026512479362137, 0.057499212591395, -0.30984396543234
    )
    val expectedOutput = Tensor[Double](Storage(expectedOutputData), 1, Array(3, 3, 3))
    val expectedGradData = Array(
      0, 0, 0, 0, 0, 0,
      -0.17478219987071, -0.061261206120279, -0.13082965455208, -0.11008777280852,
      -0.091595783867018, 0.025600875888105,
      -0.1678862752483, 0.1705562017599, -0.03868633899779, -0.044810903598228,
      -0.12107219386202, -0.081803252098247,
      0, 0, 0, 0, 0, 0,
      -0.091964358838267, -0.097584847687116, -0.18714311206719, 0.12176920040528,
      -0.1468025131371, -0.12983631153158,

      0, 0, 0, 0, 0, 0,
      -0.034294782621376, 0.047300243300225, 0.008911150511847, -0.14627057232367,
      -0.035318171790018, -0.035607346551946,
      -0.13461988398503, 0.14504585663619, -0.14020844128075, 0.039466216288395,
      -0.073137606855201, 0.069186894547888,
      0, 0, 0, 0, 0, 0,
      0.070098395440994, 0.054066545221617, 0.010560706796531, 0.033162304609837,
      -0.14235221, -0.20526800703813
    )
    val expectedGrad = Tensor[Double](Storage(expectedGradData), 1, Array(2, 5, 6))
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
    val gradBiasData = Array(
      5.1377515662462, 3.8248416814022, 2.8428408897016
    )
    val gradBias = Tensor[Double](Storage(gradBiasData), 1, Array(3))
    val gradWeightData = Array(
      2.0666666537599, 1.5350372840705, 2.5491404817346,
      0.079378455201224, 0.66457160043632, 0.3247547531408,

      1.671256460154, 2.0653377796976, 1.8402419617972,
      0.79483949649916, 0.4561988102432, 0.51269414023682,

      1.1268715925462, 0.98475658370569, 1.9118329664013,
      0.082461188620462, 0.53122743841619, 0.40517868313843,

      0.91009567143048, 1.3992566444503, 1.1852642748638,
      0.18820602302855, 0.35597275906383, 0.57617636028019,

      0.76470984306915, 0.66668800318815, 1.2236451865942,
      0.037865577254711, 0.34429103180817, 0.15072845747461,

      0.6242494305183, 1.1418853367335, 0.94981153758242,
      0.44375239087041, 0.2308612946304, 0.278725442139
    )
    val gradWeight = Tensor[Double](Storage(gradWeightData), 1, Array(3, 2, 2, 3))
    module.gradBias.map(gradBias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    module.gradWeight.map(gradWeight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }

  "A SpatialConvolution" should
    "generate correct gradWeight gradBias output gradInput called twice" in {
    val weightData = Array(
      0.076179399003951, 0.011624260825773, -0.26929373771844,
      0.21727415347489, -0.12130985860073, -0.075744581174401,

      0.13516901658924, -0.11696029209996, -0.22248883304562,
      0.023747188857974, -0.21052245873008, -0.076731029460861,

      0.27798097331555, -0.21490849912455, 0.15393807971066,
      -0.10031578297795, -0.28516644100567, -0.19091797389294,

      0.12532535545942, -0.072735155267421, -0.013593492171148,
      0.066118103528912, -0.12672802914211, 0.19073741499255,

      -0.28819224469682, -0.18149741183885, 0.036125793463643,
      0.16313411236647, 0.24139356738889, 0.12428446592142,

      0.00075597771376307, 0.24760437974888, 0.017539558871527,
      0.0097350586336657, 0.16691246244305, 0.089457577292526
    )
    val weight = Tensor[Double](Storage(weightData), 1, Array(3, 2, 2, 3))
    val biasData = Array(
      -0.2619934213478, 0.28232013358659, -0.048443703240033
    )
    val bias = Tensor[Double](Storage(biasData), 1, Array(3))
    val module = new SpatialConvolution[Double](2, 3, 3, 2, 2, 3, 1, 2)
    module.weight.copy(weight)
    module.bias.copy(bias)
    val inputData = Array(
      0.31065078778192, 0.47680206806399, 0.47333685657941, 0.60491567384452, 0.45523861795664,
      0.59684172924608,
      0.18907571188174, 0.86919780098833, 0.10920453490689, 0.32645826041698, 0.34624883672222,
      0.82086082990281,
      0.2254743387457, 0.054266506806016, 0.44735967810266, 0.18092241417617, 0.77504938514903,
      0.84614481125027,
      0.38507605646737, 0.01930161495693, 0.71951506263576, 0.51190832140855, 0.29157469817437,
      0.96515286993235,
      0.55696543189697, 0.39736612164415, 0.37251793802716, 0.94657975435257, 0.78204862540588,
      0.81618627184071,

      0.21099418401718, 0.94533503637649, 0.18719119438902, 0.439231029246, 0.6292649700772,
      0.49809367349371,
      0.84911825321615, 0.66236829245463, 0.22733291704208, 0.20995571650565, 0.20888273068704,
      0.022939298069105,
      0.026140305912122, 0.78018275159411, 0.34802454058081, 0.10564375855029, 0.68778858194128,
      0.19946397026069,
      0.22409740672447, 0.93569488637149, 0.12785892537795, 0.15872345422395, 0.59272385016084,
      0.29062455683015,
      0.41214634198695, 0.37712478893809, 0.96965333609842, 0.76970787090249, 0.4584473001305,
      0.55620927736163
    )
    val input = Tensor[Double](Storage(inputData), 1, Array(2, 5, 6))
    val gradOutputData = Array(
      0.62875230400823, 0.32400949206203, 0.34562376490794,
      0.48625181033276, 0.99999123509042, 0.13880217191763,
      0.536536491476, 0.79385008965619, 0.88393420679495,

      0.16750993672758, 0.87882893625647, 0.43605240457691,
      0.6109996503219, 0.18711599125527, 0.39965781872161,
      0.0030532660894096, 0.44951300788671, 0.69211066956632,

      0.68326092790812, 0.13290155655704, 0.019294884288684,
      0.27066777111031, 0.56331930914894, 0.040327817667276,
      0.53744582436047, 0.54968754481524, 0.045935253845528
    )
    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(3, 3, 3))
    val expectedOutputData = Array(
      -0.2619934213478, -0.2619934213478, -0.2619934213478,
      -0.83737774911124, -0.38522056607752, -0.73170056476513,
      -0.49463812878686, -0.71598681999743, -0.47391648638437,

      0.28232013358659, 0.28232013358659, 0.28232013358659,
      0.37556331587562, 0.47442123673519, -0.0070165134165954,
      0.18868933521803, 0.42471014791584, 0.55858239657379,

      -0.048443703240033, -0.048443703240033, -0.048443703240033,
      0.30583199116269, -0.032030947882614, 0.33200759191869,
      -0.026512479362137, 0.057499212591395, -0.30984396543234
    )
    val expectedOutput = Tensor[Double](Storage(expectedOutputData), 1, Array(3, 3, 3))
    val expectedGradData = Array(
      0, 0, 0, 0, 0, 0,
      -0.17478219987071, -0.061261206120279, -0.13082965455208, -0.11008777280852,
      -0.091595783867018, 0.025600875888105,
      -0.1678862752483, 0.1705562017599, -0.03868633899779, -0.044810903598228,
      -0.12107219386202, -0.081803252098247,
      0, 0, 0, 0, 0, 0,
      -0.091964358838267, -0.097584847687116, -0.18714311206719, 0.12176920040528,
      -0.1468025131371, -0.12983631153158,

      0, 0, 0, 0, 0, 0,
      -0.034294782621376, 0.047300243300225, 0.008911150511847, -0.14627057232367,
      -0.035318171790018, -0.035607346551946,
      -0.13461988398503, 0.14504585663619, -0.14020844128075, 0.039466216288395,
      -0.073137606855201, 0.069186894547888,
      0, 0, 0, 0, 0, 0,
      0.070098395440994, 0.054066545221617, 0.010560706796531, 0.033162304609837,
      -0.14235221, -0.20526800703813
    )
    val expectedGrad = Tensor[Double](Storage(expectedGradData), 1, Array(2, 5, 6))
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    module.forward(input)
    module.backward(input, gradOutput)
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      v1 should be(v2 +- 1e-6)
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
    val gradBiasData = Array(
      5.1377515662462, 3.8248416814022, 2.8428408897016
    )
    val gradBias = Tensor[Double](Storage(gradBiasData), 1, Array(3))
    val gradWeightData = Array(
      2.0666666537599, 1.5350372840705, 2.5491404817346,
      0.079378455201224, 0.66457160043632, 0.3247547531408,

      1.671256460154, 2.0653377796976, 1.8402419617972,
      0.79483949649916, 0.4561988102432, 0.51269414023682,

      1.1268715925462, 0.98475658370569, 1.9118329664013,
      0.082461188620462, 0.53122743841619, 0.40517868313843,

      0.91009567143048, 1.3992566444503, 1.1852642748638,
      0.18820602302855, 0.35597275906383, 0.57617636028019,

      0.76470984306915, 0.66668800318815, 1.2236451865942,
      0.037865577254711, 0.34429103180817, 0.15072845747461,

      0.6242494305183, 1.1418853367335, 0.94981153758242,
      0.44375239087041, 0.2308612946304, 0.278725442139
    )
    val gradWeight = Tensor[Double](Storage(gradWeightData), 1, Array(3, 2, 2, 3))
    module.gradBias.map(gradBias * 2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    module.gradWeight.map(gradWeight * 2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
  "A SpatialConvolution" should "generate correct result in actual training" in {
    val weightData = Array(
      -0.12261191252619, -0.14672847865149, -0.16658302862197, -0.14940487202257,
      -0.1492015985772,
      0.060894620697945, -0.014343269821256, 0.12106087524444, 0.0026984692551196,
      -0.025313082151115,
      0.12785792518407, 0.041174768004566, -0.0048693394288421, 0.1985639530234,
      -0.1485321813263,
      0.13702007681131, 0.053636099305004, -0.10813079550862, -0.18637271756306,
      0.034501196444035,
      -0.12549535976723, 0.18022901667282, 0.045029263570905, -0.002384402602911,
      -0.11517680492252,

      -0.015803681500256, 0.12332183504477, 0.086423795018345, -0.18231090055779,
      -0.16094405893236,
      0.072888857498765, 0.060086951032281, -0.15491142077371, 0.13384778182954,
      -0.023988491762429,
      0.13307872135192, -0.15988203156739, -0.15437647309154, 0.044045650027692,
      0.0090599200688302,
      0.1625052346848, -0.033724643103778, 0.1382389000617, 0.013144145999104,
      -0.088398958090693,
      -0.13420979883522, -0.02474349392578, -0.07446510232985, -0.06092475168407,
      -0.024276099633425
    )
    val weight = Tensor[Double](Storage(weightData), 1, Array(2, 1, 5, 5))
    val biasData = Array(
      -0.140272492636, 0.12101946240291
    )
    val bias = Tensor[Double](Storage(biasData), 1, Array(2))
    val exInputData = Array(
      1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
      1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
      1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1,
      1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2,
      1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3,
      1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4,
      1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5,
      1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
    )
    val exInput = Tensor[Double](Storage(exInputData), 1, Array(1, 8, 8))
    val exOutputData = Array(
      -0.11965132333453, -0.12005417477305, -0.12045702621157, -0.12085987765009,
      -0.12005417477305, -0.12045702621157, -0.12085987765009, -0.12126272908862,
      -0.12045702621157, -0.12085987765009, -0.12126272908862, -0.12166558052714,
      -0.12085987765009, -0.12126272908862, -0.12166558052714, -0.12206843196566,
      0.1018337093967, 0.10720521214648, 0.11257671489626, 0.11794821764604,
      0.10720521214648, 0.11257671489626, 0.11794821764604, 0.12331972039582,
      0.11257671489626, 0.11794821764604, 0.12331972039582, 0.1286912231456,
      0.11794821764604, 0.12331972039582, 0.1286912231456, 0.13406272589537
    )
    val exOutput = Tensor[Double](Storage(exOutputData), 1, Array(32))
    val exGradInputData = Array(
      0.0079654085733948, 0.0096565042347231, 0.014703831650016, 0.033518589810872,
      0.042944031100268, 0.041046080578993, 0.035838139843191, 0.017224067379238,
      -0.0020415131137044, -0.0044526439282245, -0.0010540999696902, 0.0078907911320059,
      0.028071642097233, 0.030225139669253, 0.026916248699726, 0.018021312942445,
      -0.020239787438288, -0.018399979585138, -0.0081278849429825, -0.017874178972477,
      0.027702719122479, 0.025781552257646, 0.015762045988039, 0.025473954516205,
      -0.040838832600394, -0.042895620727759, -0.035097919750491, -0.034848486899971,
      0.031159903640922, 0.03325829042838, 0.025587609494698, 0.025337021217108,
      -0.034478409423555, -0.051934185990899, -0.050775712830284, -0.068155798754743,
      -0.019012571532447, -0.0013570111465062, -0.002224675392083, 0.014932478452833,
      -0.024486688552409, -0.037931483885193, -0.034948744476006, -0.042646366717206,
      -0.004293920189748, 0.0094014373441851, 0.0064592405405205, 0.014084661054666,
      -0.0063790068156164, -0.023866312011206, -0.027596349987023, -0.016651179533407,
      -0.0037128573723662, 0.013848735793556, 0.017456380991353, 0.0065233496564268,
      0.014089544022944, 0.00053779660636034, -0.00096172856578671, -5.9256955889099e-05,
      -0.007382602101755, 0.0061204756133857, 0.0076231763364756, 0.0066996137710091
    )
    val exGradInput = Tensor[Double](Storage(exGradInputData), 1, Array(1, 8, 8))
    val exWeightData = Array(
      -0.10789623575114, -0.13057475680118, -0.1489912616964, -0.13037506002174,
      -0.12873374150111,
      0.077048342548256, 0.0032484971043147, 0.14009068724527, 0.023166326331213,
      -0.003407179999761,
      0.14544969210964, 0.060204580005398, 0.015598517647251, 0.22046985517476,
      -0.12518823409969,
      0.15604988881215, 0.074103956381097, -0.086224893357269, -0.16302877033645,
      0.05928318874591,
      -0.10502750269114, 0.20213491882417, 0.06837321079752, 0.022397589698965,
      -0.088956767545384,

      -0.0039755057591689, 0.13631193713497, 0.10057582345765, -0.16699694576938,
      -0.14446817779484,
      0.08587895958896, 0.074238979471584, -0.1395974659853, 0.15032366296706,
      -0.0063506842758023,
      0.14723074979122, -0.14456807677898, -0.13790059195402, 0.06168345751432,
      0.027859653904565,
      0.17781918947321, -0.017248761966259, 0.15587670754832, 0.031943879834839,
      -0.06843729790585,
      -0.1177339176977, -0.0071056864391529, -0.055665368494115, -0.040963091499226,
      -0.0031525130994743
    )
    val exWeight = Tensor[Double](Storage(exWeightData), 1, Array(2, 1, 5, 5))
    val exBiasData = Array(
      -0.12589204188339, 0.13263872589399
    )
    val exBias = Tensor[Double](Storage(exBiasData), 1, Array(2))
    val gradWeightData = Array(
      -1.6813905293348, -1.7934765170998, -1.9055625048648, -2.0176484926298, -2.1297344803948,
      -1.7934765170998, -1.9055625048648, -2.0176484926298, -2.1297344803948, -2.2418204681598,
      -1.9055625048648, -2.0176484926298, -2.1297344803948, -2.2418204681598, -2.3539064559248,
      -2.0176484926298, -2.1297344803948, -2.2418204681598, -2.3539064559248, -2.4659924436898,
      -2.1297344803948, -2.2418204681598, -2.3539064559248, -2.4659924436898, -2.5780784314548,

      -1.3217347978435, -1.4099399760789, -1.4981451543143, -1.5863503325497, -1.6745555107851,
      -1.4099399760789, -1.4981451543143, -1.5863503325497, -1.6745555107851, -1.7627606890205,
      -1.4981451543143, -1.5863503325497, -1.6745555107851, -1.7627606890205, -1.8509658672559,
      -1.5863503325497, -1.6745555107851, -1.7627606890205, -1.8509658672559, -1.9391710454913,
      -1.6745555107851, -1.7627606890205, -1.8509658672559, -1.9391710454913, -2.0273762237267
    )
    val gradWeight = Tensor[Double](Storage(gradWeightData), 1, Array(2, 1, 5, 5))
    val gradBiasData = Array(
      -1.1208598776501, -0.88205178235396
    )
    val gradBias = Tensor[Double](Storage(gradBiasData), 1, Array(2))
    val exErr = 1.0172073752036
    val maxIter = 10
    var model = new Sequential[Double]()
    var sc = new SpatialConvolution[Double](1, 2, 5, 5)

    sc.weight.copy(weight)
    sc.bias.copy(bias)
    model.add(sc)
    val linearInputNum = 2 * 4 * 4
    val rs = new Reshape[Double](Array(linearInputNum), Some(false))
    model.add(rs)
    val input = Tensor[Double](10, 1, 8, 8)
    for (i <- 1 to 10)
      for (j <- 1 to 8)
        for (k <- 1 to 8)
          input(Array(i, 1, j, k)) = (i + j + k) / 10.0
    val loss = new MSECriterion[Double]()
    val t = Tensor[Double](linearInputNum)
    t.fill(1)
    var err = 0.0
    var output: Tensor[Double] = null
    var gradOutput: Tensor[Double] = null
    var gradInput: Tensor[Double] = null

    val (w, g) = model.getParameters()
    for (k <- 1 to maxIter) {
      model.zeroGradParameters()
      output = model.forward(input(k)).toTensor[Double]
      err = loss.forward(output, t)
      gradOutput = loss.backward(output, t)
      gradInput = model.backward(input(k), gradOutput).toTensor[Double]
      w.add(-0.001, g)
    }

    input(maxIter).map(exInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    gradInput.map(exGradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    output.map(exOutput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    sc.gradWeight.map(gradWeight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    sc.gradBias.map(gradBias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    sc.weight.map(exWeight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    sc.bias.map(exBias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-4);
      v1
    })
  }

  "A SpatialConvolutionM" should "generate correct output" in {
    val nInputPlane = 3
    val nOutputPlane = 16
    val kW = 5
    val kH = 5
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](3, 16, kW, kH, dW, dH, padW, padH)

    val inputData = new Array[Double](3072)
    for (j <- 0 to 11) {
      for (i <- 0 to 255) {
        inputData(i + 256 * j) = i
      }
    }


    val input = Tensor[Double](Storage(inputData), 1, Array(nInputPlane, 32, 32))
    val output = layer.updateOutput(input)
  }

  "A SpatialConvolution with batch input" should "be good in gradient checker" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
    layer.reset()

    val input = Tensor[Double](2, 1, 5, 5).rand()
    val checker = new GradientChecker(1e-4, 1e-2)
    checker.checkLayer[Double](layer, input) should be(true)

  }

  "A SpatialConvolution with scaleW and scaleB" should "generate correct gradWeight gradBias" in {
    val seed = 100
    RNG.setSeed(seed)
    val nInputPlane = 3
    val nOutputPlane = 64
    val kW = 11
    val kH = 11
    val dW = 4
    val dH = 4
    val padW = 2
    val padH = 2
    val layer1 = new SpatialConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)
    val layer2 = layer1.cloneModule().asInstanceOf[SpatialConvolution[Double]]
    layer2.setScaleW(2).setScaleB(0.5)

    val input = Tensor[Double](16, 3, 224, 224).apply1(e => Random.nextDouble())

    val output1 = layer1.forward(input)
    val output2 = layer2.forward(input)
    output1 should be (output2)

    val gradOutput = Tensor[Double]().resizeAs(output1).apply1(e => Random.nextDouble())
    val gradInput1 = layer1.backward(input, gradOutput)
    val gradInput2 = layer2.backward(input, gradOutput)
    gradInput1 should be (gradInput2)

    layer2.gradWeight should be (layer1.gradWeight.mul(2))
    layer2.gradBias should be (layer1.gradBias.mul(0.5))
  }

  "A SpatialConvolution layer without bias" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH, withBias = false)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }

  "Xavier" should "init right in SpatialConvolution" in {
    RNG.setSeed(1)
    val conv = SpatialConvolution[Float](2, 4, 3, 3, 2, 2, 3, 3, 1, false)
      .setInitMethod(Xavier, Zeros)
    val exceptedWeight = Tensor[Float](Storage(Array(
    -0.32114115, -0.31055245, 0.16676287,
    0.082686655, 0.32590738, 0.10709048,
    0.16544376, -0.13433647, -0.14637068,

    -0.035910334, 0.19285288, -0.1852503,
    -0.264516, -0.2844239, -0.03473765,
    -0.02050765, 0.272397, -0.2692185,

    -0.13759057, 0.26891345, -0.1414831,
    -0.25367302, -0.24664763, 0.016532922,
    -0.32042202, -0.27758467, 0.119223684,

    0.27790755, -0.19224793, 0.27363226,
    -0.15630223, -0.1340466, -0.0056178933,
    0.056259416, -0.2977583, 0.043941353,

    0.049411736, 0.07595888, -0.23551428,
    0.3043571, 0.059537023, -0.15934734,
    0.13317224, -0.17932305, -0.26511037,

    0.022298995, -0.057296008, 0.29995877,
    0.12960011, -0.0046269377, -0.057213824,
    0.027067006, -0.30003104, 0.17699008,

    0.023930939, -0.30310285, 0.10919643,
    -0.24002258, 0.009926071, 0.19493572,
    0.2963965, -0.31346577, 0.05770336,

    0.255417, 0.2689346, 0.027192127,
    -0.24168353, -0.03467988, -0.24048243,
    0.26142392, 0.20492753, -0.081610434).map(_.toFloat))).resize(1, 4, 2, 3, 3)
    val exceptedBias = Tensor[Float](T(0f, 0f, 0f, 0f))
    conv.weight should be (exceptedWeight)
    conv.bias should be (exceptedBias)
  }

  "hashcode & clearState" should "works fine" in {
    val layer = new SpatialConvolution[Float](3, 4,
      2, 2, 1, 1, 0, 0, withBias = false)
    val input = Tensor[Float](2, 3, 4, 4).rand()
    val output = layer.forward(input).toTensor[Float]
    layer.backward(input, output.clone().rand)
    layer.hashCode()
    layer.clearState()
  }

  "equals" should "works fine" in {
    val layer = new SpatialConvolution[Float](3, 4,
      2, 2, 1, 1, 0, 0, withBias = false)
    val layer2 = layer.cloneModule()
    layer.equals(layer2) should be (true)

    val layer3 = new SpatialConvolution[Float](3, 4,
      2, 2, 1, 1, 0, 0)
    layer3.equals(layer) should be (false)
    layer3.weight.copy(layer.weight)
    layer3.equals(layer) should be (false)
  }

  "SpatialConvolution computeOutputShape NCHW" should "work properly" in {
    val layer = SpatialConvolution[Float](3, 5, 2, 2)
    TestUtils.compareOutputShape(layer, Shape(3, 12, 12)) should be (true)
  }

  "SpatialConvolution computeOutputShape NHWC" should "work properly" in {
    val layer = SpatialConvolution[Float](4, 5, 2, 2, format = DataFormat.NHWC)
    TestUtils.compareOutputShape(layer, Shape(12, 12, 4)) should be (true)
  }
}

class SpatialConvolutionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialConvolution = SpatialConvolution[Float](3, 4, 2, 2).
      setName("spatialConvolution")
    val input = Tensor[Float](1, 3, 5, 5).apply1( e => Random.nextFloat())
    runSerializationTest(spatialConvolution, input)
  }
}
