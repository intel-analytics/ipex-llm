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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.nn.{ReLU, SpatialConvolution}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class ConvolutionDnnSpec extends FlatSpec with Matchers {

  "ConvolutionDnn with format=nchw and ngroup=1" should "work correctly" in {
      val nInputPlane = 2
      val nOutputPlane = 4
      val kW = 3
      val kH = 3
      val dW = 4
      val dH = 4
      val padW = 0
      val padH = 0

      val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
      val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
      RNG.setSeed(100)
      val conv = ConvolutionDnn(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      RNG.setSeed(100)
      val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

      val output = conv.forward(input)
      val grad1 = conv.updateGradInput(input, gradOutput)
      conv.accGradParameters(input, gradOutput)
      conv.accGradParameters(input, gradOutput)
      val weight1 = conv.weight
      val gradweight1 = conv.gradWeight
      val bias1 = conv.bias
      val gradbias1 = conv.gradBias
      val output2 = layer.forward(input)
      val grad2 = layer.updateGradInput(input, gradOutput)
      layer.accGradParameters(input, gradOutput)
      layer.accGradParameters(input, gradOutput)
      val weight2 = layer.weight
      val gradweight2 = layer.gradWeight
      val bias2 = conv.bias
      val gradbias2 = conv.gradBias

      DnnUtils.nearequals(weight1, weight2) should be(true)
      DnnUtils.nearequals(gradweight1, gradweight2) should be(true)
      DnnUtils.nearequals(bias1, bias2) should be(true)
      DnnUtils.nearequals(gradbias1, gradbias2) should be(true)
      DnnUtils.nearequals(output, output2) should be(true)
      DnnUtils.nearequals(grad1, grad2) should be(true)
  }

  "ConvolutionDnn with format=nchw and ngroup=2" should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0
    val ngroup = 2

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = ConvolutionDnn(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, ngroup)
    RNG.setSeed(100)
    val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH,
      dW, dH, padW, padH, ngroup)

    val output = conv.forward(input)
    val grad1 = conv.updateGradInput(input, gradOutput)
    conv.accGradParameters(input, gradOutput)
    conv.accGradParameters(input, gradOutput)
    val weight1 = conv.weight
    val gradweight1 = conv.gradWeight
    val bias1 = conv.bias
    val gradbias1 = conv.gradBias
    val output2 = layer.forward(input)
    val grad2 = layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
    layer.accGradParameters(input, gradOutput)
    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = conv.bias
    val gradbias2 = conv.gradBias

    DnnUtils.nearequals(weight1, weight2) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2) should be(true)
    DnnUtils.nearequals(bias1, bias2) should be(true)
    DnnUtils.nearequals(gradbias1, gradbias2) should be(true)
    DnnUtils.nearequals(output, output2) should be(true)
    DnnUtils.nearequals(grad1, grad2) should be(true)
  }

  "ConvolutionDnn with relu " should "work correctly" in {
    val nInputPlane = 2
    val nOutputPlane = 4
    val kW = 3
    val kH = 3
    val dW = 4
    val dH = 4
    val padW = 0
    val padH = 0
    val ngroup = 2

    val input = Tensor[Float](2, 2, 23, 23).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](2, 4, 6, 6).apply1(e => Random.nextFloat())
    RNG.setSeed(100)
    val conv = ConvolutionDnn(nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, ngroup)
    RNG.setSeed(100)
    val layer = SpatialConvolution[Float](nInputPlane, nOutputPlane, kW, kH,
      dW, dH, padW, padH, ngroup)
    val relu = ReLUDnn[Float](ip = false)
    val relu1 = ReLU[Float](ip = false)

    var output = conv.forward(input)
    relu.forward(output)
    val grad1 = relu.backward(output, gradOutput)
    val grad1_conv = conv.backward(input, grad1)

    val weight1 = conv.weight
    val gradweight1 = conv.gradWeight
    val bias1 = conv.bias
    val gradbias1 = conv.gradBias

    val output2 = layer.forward(input)
    relu.forward(output2)
    val grad2 = relu.backward(output2, gradOutput)
    val grad2_conv = layer.backward(input, grad2)

    val weight2 = layer.weight
    val gradweight2 = layer.gradWeight
    val bias2 = conv.bias
    val gradbias2 = conv.gradBias

    DnnUtils.nearequals(weight1, weight2) should be(true)
    DnnUtils.nearequals(gradweight1, gradweight2) should be(true)
    DnnUtils.nearequals(bias1, bias2) should be(true)
    DnnUtils.nearequals(gradbias1, gradbias2) should be(true)
    DnnUtils.nearequals(output, output2) should be(true)
    DnnUtils.nearequals(grad1, grad2) should be(true)
  }
}
