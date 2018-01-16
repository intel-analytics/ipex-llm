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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class SpatialBatchNormalizationSpec extends FlatSpec with Matchers {
  "SpatialBatchNormalization module in batch mode" should "be good in gradient check " +
    "for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val sbn = new SpatialBatchNormalization[Double](3, 1e-3)
    val input = Tensor[Double](16, 3, 4, 4).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkLayer[Double](sbn, input, 1e-3) should be(true)
  }

  "SpatialBatchNormalization backward" should "be good when affine is false" in {
    val layer = SpatialBatchNormalization[Float](3, affine = false)
    val input = Tensor[Float](4, 3, 24, 24).fill(1)
    val gradOutput = Tensor[Float](4, 3, 24, 24).fill(1)
    val output = layer.forward(input)
    output should be(Tensor[Float](4, 3, 24, 24).fill(0))
    val gradInput = layer.backward(input, gradOutput)
    gradInput should be(Tensor[Float](4, 3, 24, 24).fill(0))
  }

  "SpatialBatchNormalization module in batch mode" should "be good in gradient check " +
    "for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val sbn = new SpatialBatchNormalization[Double](3, 1e-3)
    val input = Tensor[Double](16, 3, 4, 4).apply1(e => Random.nextDouble())

    val checker = new GradientChecker(1e-4)
    checker.checkWeight[Double](sbn, input, 1e-3) should be(true)
  }

  "A SpatialBatchNormalization" should "generate same output for NHWC and NCHW" in {
    val inputNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val inputNHWC = inputNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val weight = Tensor[Float](256).rand()
    val bias = Tensor[Float](256).rand()
    val bnNCHW = SpatialBatchNormalization[Float](nOutput = 256, initWeight = weight,
      initBias = bias)
    val bnNHWC = SpatialBatchNormalization[Float](nOutput = 256, dataFormat = DataFormat.NHWC,
      initWeight = weight, initBias = bias)
    val outputNCHW = bnNCHW.forward(inputNCHW)
    val outputNHWC = bnNHWC.forward(inputNHWC)
    outputNCHW.almostEqual(outputNHWC.transpose(2, 4).transpose(3, 4), 1e-5) should be(true)
  }

  "A SpatialBatchNormalization update gradinput" should
    "generate same output for NHWC and NCHW" in {
    val inputNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val inputNHWC = inputNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val gradientNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val gradientNHWC = gradientNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val weight = Tensor[Float](256).rand()
    val bias = Tensor[Float](256).rand()
    val bnNCHW = SpatialBatchNormalization[Float](nOutput = 256, initWeight = weight,
      initBias = bias)
    val bnNHWC = SpatialBatchNormalization[Float](nOutput = 256, dataFormat = DataFormat.NHWC,
      initWeight = weight, initBias = bias)
    val outputNCHW = bnNCHW.forward(inputNCHW)
    val outputNHWC = bnNHWC.forward(inputNHWC)

    val backpropNCHW = bnNCHW.updateGradInput(inputNCHW, gradientNCHW)
    val backpropNHWC = bnNHWC.updateGradInput(inputNHWC, gradientNHWC)

    backpropNCHW.almostEqual(backpropNHWC.transpose(2, 4).transpose(3, 4), 1e-5) should be(true)
  }

  "A SpatialBatchNormalization acc gradient" should "generate same output for NHWC and NCHW" in {
    val inputNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val inputNHWC = inputNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val gradientNCHW = Tensor[Float](4, 256, 8, 8).rand()
    val gradientNHWC = gradientNCHW.transpose(2, 4).transpose(2, 3).contiguous()
    val weight = Tensor[Float](256).rand()
    val bias = Tensor[Float](256).rand()
    val bnNCHW = SpatialBatchNormalization[Float](nOutput = 256, initWeight = weight,
      initBias = bias)
    val bnNHWC = SpatialBatchNormalization[Float](nOutput = 256, dataFormat = DataFormat.NHWC,
      initWeight = weight, initBias = bias)
    val outputNCHW = bnNCHW.forward(inputNCHW)
    val outputNHWC = bnNHWC.forward(inputNHWC)

    bnNCHW.backward(inputNCHW, gradientNCHW)
    bnNHWC.backward(inputNHWC, gradientNHWC)

    bnNCHW.gradWeight.almostEqual(bnNHWC.gradWeight, 1e-5)
    bnNCHW.gradBias.almostEqual(bnNHWC.gradBias, 1e-5)
  }
}

