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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.{SpatialConvolution => NNSpatialConvolution}
import com.intel.analytics.bigdl.nn.fixpoint.{SpatialConvolution => FPSpatialConvolution}

import java.io.{File, PrintWriter}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SpatialConvolutionSpec extends FlatSpec with Matchers {
  val testCases = List(
    TestCase(1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 0, 0),
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

      nn.bias.fill(0f)
      for (i <- 0 until nn.weight.nElement()) {
        nn.weight.storage().array()(i) = i % 32
      }

      fp.weight.copy(nn.weight)
      fp.bias.copy(nn.bias)

      val input = Tensor[Float]().resize(Array(test.batchSize, test.inputChannel,
        test.inputHeight, test.inputWidth)).rand()

      for (i <- 0 until input.nElement()) {
        input.storage().array()(i) = i % 32
      }

      val nnOutput = nn.updateOutput(input)
      val fpOutput = fp.updateOutput(input)

      fp.release()

      val file = s"/tmp/output/${fp.toString().filterNot((x: Char) => x.isWhitespace)}"
      Tools.writeTensor2File(fpOutput, file)

      Tools.compare2Tensors(nnOutput, fpOutput) should be (0)
    }
  }
}

object Tools {
  val magicValue = 1f

  def compare2Tensors(a1: Tensor[Float], a2: Tensor[Float]): Int = {
    var ret = true

    if (a1.nElement() != a2.nElement()) {
      ret = false
    }

    var count = 0

    for (i <- 0 until a1.nElement()) {
      val a1t = a1.storage().array()(i)
      val a2t = a2.storage().array()(i)
      if (math.abs(a1t - a2t) > magicValue) {
        count += 1
      }
    }

    println(s"total = ${a1.nElement()} count = $count")
    count
  }

  def writeTensor2File(tensor: Tensor[Float], file: String): Unit = {
    val printWriter = new PrintWriter(new File(file))
    try {
      for (i <- 0 until tensor.nElement()) {
        printWriter.write(tensor.storage().array()(i).toString + "\n")
      }
    } finally {
      printWriter.close()
    }
  }
}

case class TestCase(
  batchSize: Int, inputChannel: Int, inputHeight: Int, inputWidth: Int,
  group: Int, outputChannel: Int,
  kernelHeight: Int, kernelWidth: Int, strideHeight: Int, strideWidth: Int,
  padHeight: Int, padWidth: Int)
