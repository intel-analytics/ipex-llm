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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.{Reshape, SpatialConvolution => NNSpatialConvolution}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers, ParallelTestExecution}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class SpatialConvolutionSpec extends FlatSpec with Matchers with ParallelTestExecution {
  // Notice:
  // 1. if we set input channel more than 1, the result will be not the same
  // 2. multi groups can't work
  val testCases = List(
    TestCase(1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0),
    TestCase(1, 1, 38, 38, 1, 2, 3, 3, 1, 1, 0, 0),
    TestCase(1, 2, 38, 38, 2, 2, 3, 3, 1, 1, 0, 0),
    TestCase(2, 1, 38, 38, 1, 84, 1, 1, 1, 1, 0, 0),
    TestCase(11, 512, 7, 7, 1, 4096, 7, 7, 1, 1, 0, 0)
  )

  for (test <- testCases) {
    val start = s"A bigquant.SpatialConvolution $test"
    start should "generate the same result with nn.SpatialConvolution" in {
      val weight = Tensor(test.group, test.outputChannel / test.group,
        test.inputChannel / test.group, test.kernelHeight, test.kernelWidth).fill(1.0f)
      val bias = Tensor(test.outputChannel).fill(0f)
      val input = Tensor().resize(Array(test.batchSize, test.inputChannel,
        test.inputHeight, test.inputWidth)).fill(1.0f)

      val nnConv = NNSpatialConvolution(test.inputChannel, test.outputChannel,
        test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
        test.padHeight, test.padWidth, test.group, initWeight = weight, initBias = bias)

      println(nnConv)

      val quantizedConv = SpatialConvolution(test.inputChannel, test.outputChannel,
        test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
        test.padHeight, test.padWidth, test.group, nnConv.weight, nnConv.bias)

      nnConv.updateOutput(input)

      quantizedConv.updateOutput(input)


      nnConv.output shouldEqual quantizedConv.output

      quantizedConv.release()
    }
  }

  "A bigquant.SpatialConvolution with dynamic input size" should "work correctly" in {
    val test = testCases(1)
    val weight = Tensor(test.group, test.outputChannel / test.group,
      test.inputChannel / test.group, test.kernelHeight, test.kernelWidth).fill(1.0f)
    val bias = Tensor(test.outputChannel).fill(0f)

    val nnConv = NNSpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, initWeight = weight, initBias = bias)

    println(nnConv)

    val quantizedConv = SpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, nnConv.weight, nnConv.bias)

    for (i <- 1 until 5) {
      val input = Tensor().resize(Array(test.batchSize, test.inputChannel,
        test.inputHeight * i * 10, test.inputWidth * i * 10)).fill(1.0f)

      nnConv.updateOutput(input)
      quantizedConv.updateOutput(input)


      nnConv.output shouldEqual quantizedConv.output
    }

    quantizedConv.release()
  }

  "A bigquant.SpatialConvolution with NHWC" should "work correctly" in {
    val test = testCases(3)
    val nnConv = NNSpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, format = DataFormat.NHWC)
    nnConv.weight.fill(1.0f)
    nnConv.bias.fill(0f)

    println(nnConv)

    val quantizedConv = SpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, nnConv.weight, nnConv.bias,
      format = DataFormat.NHWC)

    for (i <- 1 until 5) {
      val input = Tensor().resize(Array(test.batchSize,
        test.inputHeight, test.inputWidth, test.inputChannel)).fill(1.0f)

      nnConv.updateOutput(input)
      quantizedConv.updateOutput(input)


      nnConv.output shouldEqual quantizedConv.output
    }

    quantizedConv.release()
  }

  "A bigquant.SpatialConvolution with 3D input" should "work correctly" in {
    val test = testCases(1)
    val weight = Tensor(test.group, test.outputChannel / test.group,
      test.inputChannel / test.group, test.kernelHeight, test.kernelWidth).fill(1.0f)
    val bias = Tensor(test.outputChannel).fill(0f)

    val nnConv = NNSpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, initWeight = weight, initBias = bias,
      format = DataFormat.NCHW)

    println(nnConv)

    val quantizedConv = SpatialConvolution(test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, nnConv.weight, nnConv.bias,
      format = DataFormat.NCHW)

    val input = Tensor().resize(Array(test.batchSize,
      test.inputHeight, test.inputWidth)).fill(1.0f)

    nnConv.updateOutput(input)
    quantizedConv.updateOutput(input)


    nnConv.output shouldEqual quantizedConv.output

    quantizedConv.release()
  }
  case class TestCase(batchSize: Int, inputChannel: Int, inputHeight: Int, inputWidth: Int,
    group: Int, outputChannel: Int, kernelHeight: Int, kernelWidth: Int,
    strideHeight: Int, strideWidth: Int, padHeight: Int, padWidth: Int)
}

class SpatialConvolutionSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val spatialConvolution = NNSpatialConvolution[Float](3, 4, 2, 2).
      setName("spatialConvolution")
    val input = Tensor[Float](1, 3, 5, 5).apply1( e => Random.nextFloat())
    runSerializationTest(spatialConvolution, input)
  }
}
