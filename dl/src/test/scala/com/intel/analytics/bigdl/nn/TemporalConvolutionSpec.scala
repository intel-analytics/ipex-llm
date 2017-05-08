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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class TemporalConvolutionSpec extends FlatSpec with Matchers {
  "A SpatialConvolution layer" should "generate correct output" in {
    val inputFrameSize = 1
    val outputFrameSize = 1
    val kW = 2
    val dW = 1
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val inputData = Array(
      1.0, 2, 3, 4, 5
    )
    val kernelData = Array(
      2.0, 3
    )
    val biasData = Array(1.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(outputFrameSize,
      kW, inputFrameSize)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(outputFrameSize)))

    val input = Tensor[Double](Storage(inputData), 1, Array(5, 1))
    val output = layer.updateOutput(input)

    output(Array(1, 1)) should be(9)
    output(Array(2, 1)) should be(14)
    output(Array(3, 1)) should be(19)
    output(Array(4, 1)) should be(24)
  }

  "A SpatialConvolution layer" should "generate correct output when inputFrameSize > 1" in {
    val inputFrameSize = 2
    val outputFrameSize = 3
    val kW = 2
    val dW = 1
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val inputData = Array(
      1.0, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20
    )
    val kernelData = Array(
      1.0, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12
    )
    val biasData = Array(1.0, 1, 1)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(outputFrameSize,
      kW * inputFrameSize)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(outputFrameSize)))

    val input = Tensor[Double](Storage(inputData), 1, Array(2, 5, 2))
    val output = layer.updateOutput(input)

    output(Array(1, 1, 1)) should be(31)
    output(Array(1, 1, 2)) should be(71)
    output(Array(1, 1, 3)) should be(111)
    output(Array(1, 2, 1)) should be(51)
    output(Array(1, 2, 2)) should be(123)
    output(Array(1, 2, 3)) should be(195)
    output(Array(1, 3, 1)) should be(71)
    output(Array(1, 3, 2)) should be(175)
    output(Array(1, 3, 3)) should be(279)
    output(Array(1, 4, 1)) should be(91)
    output(Array(1, 4, 2)) should be(227)
    output(Array(1, 4, 3)) should be(363)
    output(Array(2, 1, 1)) should be(131)
    output(Array(2, 1, 2)) should be(331)
    output(Array(2, 1, 3)) should be(531)
    output(Array(2, 2, 1)) should be(151)
    output(Array(2, 2, 2)) should be(383)
    output(Array(2, 2, 3)) should be(615)
    output(Array(2, 3, 1)) should be(171)
    output(Array(2, 3, 2)) should be(435)
    output(Array(2, 3, 3)) should be(699)
    output(Array(2, 4, 1)) should be(191)
    output(Array(2, 4, 2)) should be(487)
    output(Array(2, 4, 3)) should be(783)
  }

  "A SpatialConvolution" should "generate correct gradInput" in {
    val inputFrameSize = 1
    val outputFrameSize = 1
    val kW = 2
    val dW = 1
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val inputData = Array(
      1.0, 2, 3, 4, 5
    )
    val kernelData = Array(
      2.0, 3
    )
    val gradOutputData = Array(
      1.0, 2, 3, 4
    )
    val biasData = Array(1.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(outputFrameSize, inputFrameSize * kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(outputFrameSize)))

    val input = Tensor[Double](Storage(inputData), 1, Array(5, 1))
    layer.updateOutput(input)

    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(4, 1))
    val gradInput = layer.updateGradInput(input, gradOutput)

    gradInput(Array(1, 1)) should be(2)
    gradInput(Array(2, 1)) should be(7)
    gradInput(Array(3, 1)) should be(12)
    gradInput(Array(4, 1)) should be(17)
    gradInput(Array(5, 1)) should be(12)
  }

  "A SpatialConvolution layer" should "generate correct gradInput when inputFrameSize > 1" in {
    val inputFrameSize = 2
    val outputFrameSize = 3
    val kW = 2
    val dW = 1
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val inputData = Array(
      1.0, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20
    )
    val kernelData = Array(
      1.0, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12
    )
    val gradOutputData = Array(
      1.0, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
      17, 18, 19, 20,
      21, 22, 23, 24
    )
    val biasData = Array(1.0, 1, 1)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(outputFrameSize,
      kW * inputFrameSize)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(outputFrameSize)))

    val input = Tensor[Double](Storage(inputData), 1, Array(2, 5, 2))
    layer.updateOutput(input)

    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(2, 4, 3))
    val gradInput = layer.updateGradInput(input, gradOutput)

    gradInput(Array(1, 1, 1)) should be(38)
    gradInput(Array(1, 1, 2)) should be(44)
    gradInput(Array(1, 2, 1)) should be(133)
    gradInput(Array(1, 2, 2)) should be(154)
    gradInput(Array(1, 3, 1)) should be(241)
    gradInput(Array(1, 3, 2)) should be(280)
    gradInput(Array(1, 4, 1)) should be(349)
    gradInput(Array(1, 4, 2)) should be(406)
    gradInput(Array(1, 5, 1)) should be(239)
    gradInput(Array(1, 5, 2)) should be(272)
    gradInput(Array(2, 1, 1)) should be(218)
    gradInput(Array(2, 1, 2)) should be(260)
    gradInput(Array(2, 2, 1)) should be(565)
    gradInput(Array(2, 2, 2)) should be(658)
    gradInput(Array(2, 3, 1)) should be(673)
    gradInput(Array(2, 3, 2)) should be(784)
    gradInput(Array(2, 4, 1)) should be(781)
    gradInput(Array(2, 4, 2)) should be(910)
    gradInput(Array(2, 5, 1)) should be(491)
    gradInput(Array(2, 5, 2)) should be(560)
  }

  "A SpatialConvolution" should "generate correct gradWeight and gradBias" in {
    val inputFrameSize = 1
    val outputFrameSize = 1
    val kW = 2
    val dW = 1
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val inputData = Array(
      1.0, 2, 3, 4, 5
    )
    val kernelData = Array(
      2.0, 3
    )
    val gradOutputData = Array(
      1.0, 2, 3, 4
    )
    val biasData = Array(1.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1,
      Array(outputFrameSize, inputFrameSize * kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(outputFrameSize)))

    val input = Tensor[Double](Storage(inputData), 1, Array(5, 1))
    layer.updateOutput(input)

    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(4, 1))

    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1)) should be(30)
    layer.gradWeight(Array(1, 2)) should be(40)

    layer.gradBias(Array(1)) should be(10)
  }

  "A SpatialConvolution layer" should "generate correct gradWeight" +
    " and gradBias when inputFrameSize > 1" in {
    val inputFrameSize = 2
    val outputFrameSize = 3
    val kW = 2
    val dW = 1
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val inputData = Array(
      1.0, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20
    )
    val kernelData = Array(
      1.0, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12
    )
    val gradOutputData = Array(
      1.0, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
      17, 18, 19, 20,
      21, 22, 23, 24
    )
    val biasData = Array(1.0, 1, 1)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(outputFrameSize,
      kW * inputFrameSize)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(outputFrameSize)))

    val input = Tensor[Double](Storage(inputData), 1, Array(2, 5, 2))
    layer.updateOutput(input)

    val gradOutput = Tensor[Double](Storage(gradOutputData), 1, Array(2, 4, 3))
    layer.updateGradInput(input, gradOutput)
    layer.accGradParameters(input, gradOutput)

    layer.gradWeight(Array(1, 1)) should be(1128)
    layer.gradWeight(Array(1, 2)) should be(1220)
    layer.gradWeight(Array(1, 3)) should be(1312)
    layer.gradWeight(Array(1, 4)) should be(1404)
    layer.gradWeight(Array(2, 1)) should be(1200)
    layer.gradWeight(Array(2, 2)) should be(1300)
    layer.gradWeight(Array(2, 3)) should be(1400)
    layer.gradWeight(Array(2, 4)) should be(1500)
    layer.gradWeight(Array(3, 1)) should be(1272)
    layer.gradWeight(Array(3, 2)) should be(1380)
    layer.gradWeight(Array(3, 3)) should be(1488)
    layer.gradWeight(Array(3, 4)) should be(1596)

    layer.gradBias(Array(1)) should be(92)
    layer.gradBias(Array(2)) should be(100)
    layer.gradBias(Array(3)) should be(108)
  }
}
