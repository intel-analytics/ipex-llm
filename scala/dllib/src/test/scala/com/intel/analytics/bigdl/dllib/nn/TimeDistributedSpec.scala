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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

class TimeDistributedSpec extends FlatSpec with Matchers {
  "A TimeDistributed Module " should "generate correct output and grad for Linear in 3D input " +
    "along first dimension" in {
    RNG.setSeed(100)

    val batchSize = 5
    val times = 5
    val inputDim = 3
    val outputDim = 4
    val timeDim = 1

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear1 = Linear[Float](inputDim, outputDim)
    val linear2 = Linear[Float](inputDim, outputDim)
    linear2.weight.map(linear1.weight, (a, b) => {b})
    linear2.bias.map(linear1.bias, (a, b) => {b})
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](linear1))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = linear2.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be (expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = linear2.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be (expectedOut)
      i += 1
    }
  }

  "A TimeDistributed Module " should "generate correct output and grad for Linear in 3D input " +
    "along second dimension" in {
    RNG.setSeed(100)

    val batchSize = 5
    val times = 3
    val inputDim = 3
    val outputDim = 4
    val timeDim = 2

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear1 = Linear[Float](inputDim, outputDim)
    val linear2 = Linear[Float](inputDim, outputDim)
    linear2.weight.map(linear1.weight, (a, b) => {b})
    linear2.bias.map(linear1.bias, (a, b) => {b})
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](linear1))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = linear2.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be (expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = linear2.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be (expectedOut)
      i += 1
    }
  }

  "A TimeDistributed Module " should "generate correct output and grad for logSoftMax " +
    "when time dimension is 2" in {
    RNG.setSeed(100)

    val batchSize = 5
    val times = 2
    val inputDim = 4
    val outputDim = 4
    val timeDim = 2

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val logSoftMax1 = LogSoftMax[Float]()
    val logSoftMax2 = LogSoftMax[Float]()
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](logSoftMax1))

    val output = model.forward(input).toTensor[Float].clone
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = logSoftMax2.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be (expectedOut)
      val expectedGradInput = logSoftMax2.backward(
        input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be (expectedGradInput)
      i += 1
    }
  }

  "A TimeDistributed Module " should "getParameters correct for linear " in {
    RNG.setSeed(100)

    val batchSize = 5
    val times = 5
    val inputDim = 3
    val outputDim = 4
    val timeDim = 1

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear1 = Linear[Float](inputDim, outputDim)
    val linear2 = Linear[Float](inputDim, outputDim)
    linear2.weight.map(linear1.weight, (a, b) => {
      b
    })
    linear2.bias.map(linear1.bias, (a, b) => {
      b
    })
    val model = Sequential[Float]()
      .add(TimeDistributed[Float](linear1))

    val (weight, grad) = model.parameters()
    val (weight2, grad2) = linear2.parameters()
    weight should be(weight2)
    grad should be(grad2)
  }
}
