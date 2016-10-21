/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class PowerSpec extends FlatSpec with Matchers {
  "A SpatialConvolution layer" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Double](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val kernelData = Array(
      2.0, 3,
      4, 5
    )

    val biasData = Array(0.0)

    layer.weight.copy(Tensor[Double](Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    layer.bias.copy(Tensor[Double](Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }

  "A Power" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0, 4, 9, 16, 25, 36)), 1, Array(2, 3))

    val power = new Power[Double](2)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power with scale" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(4.0, 16, 36, 64, 100, 144)), 1, Array(2, 3))

    val power = new Power[Double](2, 2)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power with shift" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0, 4, 9, 16, 25, 36)), 1, Array(2, 3))

    val power = new Power[Double](2, 1, 1)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power with scale and shift" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0, 9, 25, 49, 81, 121)), 1, Array(2, 3))

    val power = new Power[Double](2, 2, 1)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power" should "generate correct grad" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(1.5, 4.5, 9.5, 16.5, 25.5, 36.5)), 1, Array(2, 3))

    val power = new Power[Double](2)

    val powerOutput = power.forward(input)
    val powerGradOutput = power.backward(input, gradOutput)

    powerGradOutput should be (gradOutput)
  }

}
