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
// * limitations under the License.
 */
// scalastyle:off
package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.scalatest.FunSuite

import scala.util.Random

class LocallyConnected1DSuite extends FunSuite {

  test("Spatial Convolution") {

    val inputFrameSize = 10
    val outputFrameSize = 8
    val kW = 5
    val dW = 2
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    val input = Tensor[Double](100, 10).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](48, 8).apply1(e => Random.nextDouble())

    val output = layer.accGradParameters(input,gradOutput)
  }

  test("locallyconnected accGradParameters") {

    val inputFrameSize = 10
    val outputFrameSize = 8
    val kW = 5
    val dW = 2
    val layer = LocallyConnected1D[Double](48,inputFrameSize, outputFrameSize, kW, dW)

    val input = Tensor[Double](100, 10).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](48, 8).apply1(e => Random.nextDouble())

    val output = layer.accGradParameters(input,gradOutput)
  }

  test("locallyconnected gradoutput") {

    val inputFrameSize = 10
    val outputFrameSize = 8
    val kW = 5
    val dW = 2
    val layer = LocallyConnected1D[Double](48,inputFrameSize, outputFrameSize, kW, dW)

    val input = Tensor[Double](100, 10).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](48, 8).apply1(e => Random.nextDouble())

    val output = layer.updateGradInput(input,gradOutput)
  }

  test("locally connected 1d") {

    val inputFrameSize = 10
    val outputFrameSize = 8
    val kW = 5
    val dW = 2
    val layer = LocallyConnected1D[Double](48, inputFrameSize, outputFrameSize, kW, dW)

    val input = Tensor[Double](100, 10).apply1(e => Random.nextDouble())
    val output = layer.updateOutput(input)

    println(output)

  }

  test("dimension and input and window") {

    val kernelW = 5
    val strideW = 2
    val outputFrameStride = (kernelW - 1) / strideW + 1
    val inputFrameStride = outputFrameStride * strideW
    //val inputoffset = j * strideW * input.size(dimFeat)
    //val outputoffset = output.storageOffset() + j * output.size(dimFeat)
    val tensor = Tensor(100, 10)
    for (i <- 0 to 999) {
      val row = i / 10 + 1
      val col = i % 10 + 1
      tensor.setValue(row, col, i)
    }

    val window = Tensor()

    val inputoffset = strideW * 10
    // val outputoffset = j * 8
    //  println(tensor)
    //  println(window.set(tensor.storage(), storageOffset = 1, Array(1, 50), strides = Array(1, 1)))
    //  println(window.set(tensor.storage(), storageOffset = 21, Array(1, 50), strides = Array(1, 1)))

    val weight2 = Tensor(8, 50)
    for (i <- 0 to 399) {
      val row = i / 50 + 1
      val col = i % 50 + 1
      weight2.setValue(row, col, i)
    }

    val windowWeight2 = Tensor()
    // println(weight2)
    //  println(window.set(weight2.storage(), storageOffset = 1, Array(8, 50), strides = Array(50, 1)))


    val d1l = 48
    val d2l = 8
    val d3l = 50
    val weight = Tensor(d1l, d2l, d3l)
    val page = d2l * d3l
    for (i <- 0 to d1l * d2l * d3l - 1) {
      val d1 = i / page + 1
      val d2 = (i % page) / (d3l) + 1
      val d3 = (i % page) % d3l + 1
      weight.setValue(d1, d2, d3, i)
    }
    print("weight")
    // println(weight)

    val windowweight = Tensor()
    //  println(windowweight.set(weight.storage(), storageOffset = 1, Array(8, 50), strides = Array(50, 1)))
    //  println(windowweight.set(weight.storage(), storageOffset = 401, Array(8, 50), strides = Array(50, 1)))

    //println(window.set(weight.storage(), storageOffset = 1, Array(8, 50, 1), strides = Array(page +60, 1, 1)))
    //println(window.set(weight.storage(), storageOffset = 21, Array(8, 50, 1), strides = Array(page + 60, 1, 1)))

    //    println(tensor)
    //    val window = Tensor()
    //    println(window.set(tensor.storage(), storageOffset = 21, Array(16, 50), strides = Array(60, 1)))
    //   println(tensor.narrow(2, 2, 2))
  }


  test("temporalConvolution") {
    val seed = 100
    RNG.setSeed(seed)

    val inputFrameSize = 10
    val outputFrameSize = 8
    val kW = 5
    val dW = 2
    val layer = TemporalConvolution[Double](inputFrameSize, outputFrameSize, kW, dW)

    Random.setSeed(seed)
    val input = Tensor[Double](100, 10).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](48, 8).apply1(e => Random.nextDouble())

    val output = layer.updateOutput(input)
    val gradInput = layer.updateGradInput(input, gradOutput)

    //  println(output)

  }


}
