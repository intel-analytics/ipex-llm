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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class SpatialDilatedConvolutionSpec extends FlatSpec with Matchers {
  "A SpatialDilatedConvolution" should "work correctly" in {
    val test = TestCase(1, 2, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0, 2, 2)

    val nnConv = nn.SpatialDilatedConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.dilatedW, test.dilatedH)
    nnConv.weight.fill(1.0f)
    nnConv.bias.fill(0.0f)
    val quantizedConv = SpatialDilatedConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.dilatedW, test.dilatedH,
      initWeight = nnConv.weight, initBias = nnConv.bias)

    val input = Tensor[Float](test.batchSize, test.inputChannel, test.inputWidth,
      test.inputHeight).fill(1.0f)

    val nnConv2 = nn.SpatialConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth)
    nnConv2.weight.fill(1.0f)
    nnConv2.bias.fill(0.0f)

    val output1 = nnConv.forward(input)
    val output2 = quantizedConv.forward(input)
    output1 should be (output2)
  }

  case class TestCase(batchSize: Int, inputChannel: Int, inputHeight: Int, inputWidth: Int,
    group: Int, outputChannel: Int, kernelHeight: Int, kernelWidth: Int,
    strideHeight: Int, strideWidth: Int, padHeight: Int, padWidth: Int, dilatedW: Int,
    dilatedH: Int)
}

class SpatialDilatedConvolutionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialDilatedConvolution = nn.SpatialDilatedConvolution[Float](1, 1,
      2, 2, 1, 1, 0, 0).setName("spatialDilatedConvolution")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialDilatedConvolution, input)
  }
}
