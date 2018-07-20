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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

class ReLUSpec extends FlatSpec with Matchers {
  "a simple relu" should "be correct" in {
    val layer = ReLU(0.0f)
    val input = Tensor[Float](T(
      T(1.0, 2.0),
      T(-1.0, -2.0)
    ))
    val seq = Sequential()
    seq.add(ReorderMemory(HeapData(Array(2, 2), Memory.Format.nc),
      HeapData(Array(2, 2), Memory.Format.nc)))
    seq.add(layer)
    seq.add(ReorderMemory(HeapData(Array(2, 2), Memory.Format.nc),
      HeapData(Array(2, 2), Memory.Format.nc)))
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(2, 2), Memory.Format.nc)))
    seq.forward(input) should be(Tensor[Float](T(
      T(1.0, 2.0),
      T(0.0, 0.0)
    )))
    val grad = Tensor[Float](T(
      T(-1.0, -2.0),
      T(1.0, 2.0)
    ))
    seq.backward(input, grad) should be(Tensor[Float](T(
      T(-1.0, -2.0),
      T(0.0, 0.0)
    )))
  }

  "Relu dnn should be same with bigdl relu" should "work correctly" in {
    val input = Tensor(4, 96, 55, 55).rand(-1, 1)
    val gradOutput = Tensor(4, 96, 55, 55).rand(-1, 1)

    val relu = nn.ReLU(ip = false)
    val reludnn = ReLU()
    val defaultFormat = HeapData(input.size(), Memory.Format.nchw)
    reludnn.setRuntime(new MklDnnRuntime)
    reludnn.initFwdPrimitives(Array(defaultFormat), TrainingPhase)
    reludnn.initBwdPrimitives(Array(defaultFormat), TrainingPhase)

    val output = relu.forward(input)
    val gradInput = relu.backward(input, gradOutput)

    val outputdnn = reludnn.forward(input)
    val gradInputdnn = reludnn.backward(input, gradOutput)

    Equivalent.nearequals(output, Tools.dense(outputdnn).toTensor) should be(true)
    Equivalent.nearequals(gradInput, Tools.dense(gradInputdnn).toTensor) should be(true)
  }

  "relu with java serialization" should "work correctly" in {
    val shape = Array(4, 96, 55, 55)
    val input = Tensor(shape).rand(-1, 1)
    val gradOutput = Tensor(shape).rand(-1, 1)

    val relu = ReLU()
    relu.setRuntime(new MklDnnRuntime)
    relu.initFwdPrimitives(Array(HeapData(shape, Memory.Format.nchw)), TrainingPhase)
    relu.initBwdPrimitives(Array(HeapData(shape, Memory.Format.nchw)), TrainingPhase)

    val cloned = SerializationUtils.clone(relu)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(HeapData(shape, Memory.Format.nchw)), TrainingPhase)
    cloned.initBwdPrimitives(Array(HeapData(shape, Memory.Format.nchw)), TrainingPhase)

    relu.forward(input)
    cloned.forward(input)

    Tools.dense(relu.output) should be (Tools.dense(cloned.output))

    relu.backward(input, gradOutput)
    cloned.backward(input, gradOutput)

    Tools.dense(relu.gradInput) should be (Tools.dense(cloned.gradInput))
  }
}
