/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{BatchNormalization, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class InternalTimeDistributedSpec extends FlatSpec with Matchers {
  "A InternalTimeDistributed Module" should "setExtraParam works correctly" in {
    RNG.setSeed(100)
    val batchSize = 5
    val times = 5
    val channels = 3
    val timeDim = 1
    val input1 = Tensor[Float](Array(batchSize, times, channels)).randn()
    val gradOutput1 = Tensor[Float](Array(batchSize, times, channels)).randn()
    val input2 = Tensor[Float](Array(batchSize, times, channels)).randn()
    val gradOutput2 = Tensor[Float](Array(batchSize, times, channels)).randn()
    val bnorm1 = BatchNormalization[Float](channels)
    val bnorm2 = BatchNormalization[Float](channels)
    val model1 = InternalTimeDistributed[Float](bnorm1
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]])
    val model2 = InternalTimeDistributed[Float](bnorm2
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]])

    model1.forward(input1)
    model1.backward(input1, gradOutput1)

    model2.forward(input2)
    model2.backward(input2, gradOutput2)

    model1.setExtraParameter(
      model2.asInstanceOf[AbstractModule[Activity, Activity, Float]].getExtraParameter())

    bnorm1.runningMean should be(bnorm2.runningMean)
    bnorm1.runningVar should be(bnorm2.runningVar)
  }

  "A InternalTimeDistributed Module" should "reset correctly" in {
    RNG.setSeed(100)
    val batchSize = 5
    val times = 5
    val inputDim = 3
    val outputDim = 4
    val timeDim = 1
    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val linear = Linear[Float](inputDim, outputDim)
    val model = InternalTimeDistributed[Float](linear
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]])

    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)

    model.reset()

    gradOutput should not be (null)
    gradInput should not be (null)
  }

  "A InternalTimeDistributed Module" should "hash code correctly" in {
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
    val model1 = Sequential[Float]()
      .add(InternalTimeDistributed[Float](linear1
        .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]))
    val model2 = Sequential[Float]()
      .add(InternalTimeDistributed[Float](linear2
        .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]))
    val hashCode1 = model1.hashCode()
    val hashCode2 = model2.hashCode()
    hashCode1 should be(hashCode2)
  }
  "A InternalTimeDistributed Module" should "getParaemtersTable correctly" in {
    RNG.setSeed(100)

    val batchSize = 5
    val times = 5
    val inputDim = 3
    val outputDim = 4
    val timeDim = 1

    val input = Tensor[Float](Array(batchSize, times, inputDim)).randn()
    val linear = Linear[Float](inputDim, outputDim)
    val model = InternalTimeDistributed[Float](linear
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]])

    model.getParametersTable() should be(linear.getParametersTable())
  }

  "A InternalTimeDistributed Module " should
    "generate correct output and grad for Linear in 3D input " +
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
    linear2.weight.map(linear1.weight, (a, b) => {
      b
    })
    linear2.bias.map(linear1.bias, (a, b) => {
      b
    })
    val model = Sequential[Float]()
      .add(InternalTimeDistributed[Float](linear1
        .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = linear2.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be(expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = linear2.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be(expectedOut)
      i += 1
    }
  }

  "A InternalTimeDistributed Module " should
    "generate correct output and grad for Linear in 3D input " +
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
    linear2.weight.map(linear1.weight, (a, b) => {
      b
    })
    linear2.bias.map(linear1.bias, (a, b) => {
      b
    })
    val model = Sequential[Float]()
      .add(InternalTimeDistributed[Float](linear1
        .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]))

    val output = model.forward(input).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = linear2.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be(expectedOut)
      i += 1
    }

    val gradOutput = Tensor[Float](Array(batchSize, times, outputDim)).randn()
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    i = 1
    while (i <= times) {
      val expectedOut = linear2.backward(input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be(expectedOut)
      i += 1
    }
  }

  "A InternalTimeDistributed Module " should
    "generate correct output and grad for logSoftMax " +
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
      .add(InternalTimeDistributed[Float](logSoftMax1
        .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]))

    val output = model.forward(input).toTensor[Float].clone
    val gradInput = model.backward(input, gradOutput).toTensor[Float].clone
    var i = 1
    while (i <= times) {
      val expectedOut = logSoftMax2.forward(input.select(timeDim, i))
      output.select(timeDim, i) should be(expectedOut)
      val expectedGradInput = logSoftMax2.backward(
        input.select(timeDim, i), gradOutput.select(timeDim, i))
      gradInput.select(timeDim, i) should be(expectedGradInput)
      i += 1
    }
  }

  "A InternalTimeDistributed Module " should "getParameters correct for linear " in {
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
      .add(InternalTimeDistributed[Float](linear1
        .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]))

    val (weight, grad) = model.parameters()
    val (weight2, grad2) = linear2.parameters()
    weight should be(weight2)
    grad should be(grad2)
  }
}

class InternalTimeDistributedSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val InternalTimeDistributed = new InternalTimeDistributed[Float](Linear[Float](5, 5)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]).
      setName("InternalTimeDistributed")
    val input = Tensor[Float](2, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(
      InternalTimeDistributed, input)
  }
}

