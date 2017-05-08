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

import com.intel.analytics.bigdl.fixpoint.FixPoint
import com.intel.analytics.bigdl.nn.{SpatialConvolution => NNSpatialConvolution}
import com.intel.analytics.bigdl.nn.fixpoint.{SpatialConvolution => FPSpatialConvolution}
import com.intel.analytics.bigdl.tensor.Tensor

case class TestCase(
  batchSize: Int, inputChannel: Int, inputHeight: Int, inputWidth: Int,
  group: Int, outputChannel: Int,
  kernelHeight: Int, kernelWidth: Int, strideHeight: Int, strideWidth: Int,
  padHeight: Int, padWidth: Int)

object Test {

  val iterations = 10

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

  def perfOne(conv: NNSpatialConvolution[Float], input: Tensor[Float]): Double = {
    for (i <- 0 until 10) {
      conv.forward(input)
    }

    val start = System.nanoTime()
    for (i <- 0 until iterations) {
      conv.forward(input)
    }
    val end = System.nanoTime()
    (end - start) / 1e6
  }

  def perfAll(testCases: List[TestCase]): Unit = {
    for (test <- testCases) {
      // nn
      val nn = new NNSpatialConvolution[Float](test.inputChannel, test.outputChannel,
        test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
        test.padHeight, test.padWidth, test.group)
      val fp = new FPSpatialConvolution[Float](test.inputChannel, test.outputChannel,
        test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
        test.padHeight, test.padWidth, test.group)

      val input = Tensor[Float]().resize(Array(test.batchSize, test.inputChannel,
        test.inputHeight, test.inputWidth)).randn()

      val nnCosts = perfOne(nn, input)
      val fpCosts = perfOne(fp, input)

      println(s"$nnCosts, $fpCosts")

      fp.release()
    }
  }
  def main(args: Array[String]): Unit = {
//    perfAll(testCases)
    FixPoint.printHello()
  }
}
