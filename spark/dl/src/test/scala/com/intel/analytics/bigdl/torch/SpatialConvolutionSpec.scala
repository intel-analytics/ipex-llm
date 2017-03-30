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

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Sequential, SpatialConvolution}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SpatialConvolutionSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A SpatialConvolution" should "generate correct output" in {
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
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)

    Random.setSeed(seed)
    val input = Tensor[Double](16, 3, 224, 224).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialConvolutionMM(3, 64, 11, 11, 4, 4, 2, 2)\n" +
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

    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput
  }


  "A SpatialConvolution(64,192,5,5,1,1,2,2)" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 64
    val nOutputPlane = 192
    val kW = 5
    val kH = 5
    val dW = 1
    val dH = 1
    val padW = 2
    val padH = 2
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH)
    val model = new Sequential[Double]()
    model.add(layer)

    Random.setSeed(3)
    val input = Tensor[Double](8, 64, 27, 27).apply1(e => Random.nextDouble())

    val output = model.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """layer = nn.SpatialConvolution(64,192,5,5,1,1,2,2)
      model = nn.Sequential()
      model:add(layer)
      weight = layer.weight
      bias = layer.bias
      model:zeroGradParameters()
      output = model:forward(input) """

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input), Array("weight", "bias",
      "output", "model"))

    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaModel = torchResult("model").asInstanceOf[Module[Double]]

    val weight = layer.weight
    val bias = layer.bias

    weight should be equals luaWeight
    bias should be equals luaBias
    output should be equals luaOutput

  }

  "A SpatialConvolution" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val layer = new SpatialConvolution[Double](3, 6, 5, 5, 1, 1, 0, 0)
    val input = Tensor[Double](3, 32, 32).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(layer, input, 1e-3) should be(true)
  }

  "A SpatialConvolution" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val layer = new SpatialConvolution[Double](3, 6, 5, 5, 1, 1, 0, 0)
    val input = Tensor[Double](3, 32, 32).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(layer, input, 1e-3) should be(true)
  }
}
