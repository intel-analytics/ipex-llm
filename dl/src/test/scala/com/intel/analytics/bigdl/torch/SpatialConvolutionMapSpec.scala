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

import com.intel.analytics.bigdl.nn.SpatialConvolutionMap
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SpatialConvolutionMapSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A SpatialConvolution" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 16
    val kW = 5
    val kH = 5
    val layer = new SpatialConvolutionMap[Double](
      SpatialConvolutionMap.random[Double](nInputPlane, nOutputPlane, 1), kW, kH)

    Random.setSeed(seed)
    val input = Tensor[Double](16, 3, 32, 32).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5)\n" +
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

  "A SpatialConvolution 3D input" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)

    val nInputPlane = 3
    val nOutputPlane = 16
    val kW = 5
    val kH = 5
    val layer = new SpatialConvolutionMap[Double](
      SpatialConvolutionMap.random[Double](nInputPlane, nOutputPlane, 1), kW, kH)

    Random.setSeed(seed)
    val input = Tensor[Double](3, 32, 32).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5)\n" +
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

}
