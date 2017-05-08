/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.TemporalConvolution
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class TemporalConvolutionSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A SpatialConvolution" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)

    val inputFrameSize = 10
    val outputFrameSize = 8
    val kW = 5
    val dW = 2
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    Random.setSeed(seed)
    val input = Tensor[Double](10, 100, 10).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "layer = nn.TemporalConvolution(10, 8, 5, 2)\n" +
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

  "A SpatialConvolution" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val layer = TemporalConvolution[Double](10, 8, 5, 2)
    val input = Tensor[Double](10, 100, 10).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(layer, input, 1e-3) should be(true)
  }

  "A SpatialConvolution" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val layer = TemporalConvolution[Double](10, 8, 5, 2)
    val input = Tensor[Double](10, 100, 10).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(layer, input, 1e-3) should be(true)
  }
}
