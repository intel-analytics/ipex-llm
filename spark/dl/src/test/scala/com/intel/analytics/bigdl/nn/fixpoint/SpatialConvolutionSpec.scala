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

package com.intel.analytics.bigdl.nn.fixpoint

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

import com.intel.analytics.bigdl.nn.{SpatialConvolution => NNSpatialConvolution}
import com.intel.analytics.bigdl.nn.fixpoint.{SpatialConvolution => FPSpatialConvolution}

@com.intel.analytics.bigdl.tags.Parallel
class SpatialConvolutionSpec extends FlatSpec with Matchers {
/*  "A SpatialConvolution layer" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      1.0f, 2f, 3f,
      4f, 5f, 6f,
      7f, 8f, 9f
    )

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f)

    layer.weight.copy(Tensor(Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    layer.bias.copy(Tensor(Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor(Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    println(output)

    layer.release()

    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }*/

  val testCases = List(
    // TestCase(1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0)
    TestCase(2, 1024, 19, 19, 1, 1024, 1, 1, 1, 1, 0, 0),
    TestCase(2, 1024, 19, 19, 1, 126, 3, 3, 1, 1, 1, 1),
    TestCase(2, 1024, 19, 19, 1, 24, 3, 3, 1, 1, 1, 1),
    TestCase(2, 256, 1, 1, 1, 16, 3, 3, 1, 1, 1, 1),
    TestCase(2, 256, 1, 1, 1, 84, 3, 3, 1, 1, 1, 1),
    TestCase(2, 256, 3, 3, 1, 16, 3, 3, 1, 1, 1, 1),
    TestCase(2, 256, 3, 3, 1, 84, 3, 3, 1, 1, 1, 1),
    TestCase(2, 256, 5, 5, 1, 126, 3, 3, 1, 1, 1, 1),
    TestCase(2, 256, 5, 5, 1, 24, 3, 3, 1, 1, 1, 1),
    TestCase(2, 512, 10, 10, 1, 126, 3, 3, 1, 1, 1, 1),
    TestCase(2, 512, 10, 10, 1, 24, 3, 3, 1, 1, 1, 1),
    TestCase(2, 512, 38, 38, 1, 16, 3, 3, 1, 1, 1, 1),
    TestCase(2, 512, 38, 38, 1, 84, 3, 3, 1, 1, 1, 1)
  )

  for (test <- testCases) {
    val start = s"A fixpoint.SpatialConvolution $test"
    println(start)
    start should "generate the same result with nn.SpatialConvolution" in {
      val nn = new NNSpatialConvolution[Float](test.inputChannel, test.outputChannel,
        test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
        test.padHeight, test.padWidth, test.group)
      val fp = new FPSpatialConvolution[Float](test.inputChannel, test.outputChannel,
        test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
        test.padHeight, test.padWidth, test.group)

      nn.reset()
      fp.reset()
      fp.weight.copy(nn.weight)
      fp.bias.copy(fp.bias)

      val input = Tensor[Float]().resize(Array(test.batchSize, test.inputChannel,
        test.inputHeight, test.inputWidth)).rand()

      val nnOutput = nn.updateOutput(input)
      val fpOutput = fp.updateOutput(input)

      fp.release()

      println(nn)
      println(fp)

      Tools.compare2Tensors(nnOutput, fpOutput) should be (true)
    }
  }
}

object Tools {
  def compare2Tensors(a1: Tensor[Float], a2: Tensor[Float]): Boolean = {
    var ret = true

    if (a1.nElement() != a2.nElement()) {
      ret = false
    }

    for (i <- 0 until a1.nElement()) {
      if (a1.storage().array()(i) != a2.storage().array()(i)) {
        ret = false
      }
    }

    return ret
  }
}

case class TestCase(
  batchSize: Int, inputChannel: Int, inputHeight: Int, inputWidth: Int,
  group: Int, outputChannel: Int,
  kernelHeight: Int, kernelWidth: Int, strideHeight: Int, strideWidth: Int,
  padHeight: Int, padWidth: Int)
