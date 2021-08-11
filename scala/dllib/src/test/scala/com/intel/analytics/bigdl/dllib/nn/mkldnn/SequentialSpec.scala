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

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import org.apache.commons.lang3.SerializationUtils

class SequentialSpec extends BigDLSpecHelper {
  "Sequential" should "not be called add after compilation" in {
    val layer = ReorderMemory(NativeData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory(NativeData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.nc))
    val seq = new Sequential()
    seq.add(layer)
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(3, 4), Memory.Format.nc)))
    intercept[IllegalArgumentException] {
      seq.add(layer2)
    }
  }

  "Sequential" should "be correct when no memory reorder happened" in {
    val layer1 = ReorderMemory(NativeData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory(NativeData(Array(3, 4), Memory.Format.io),
      NativeData(Array(3, 4), Memory.Format.nc))
    val layer3 = ReorderMemory(HeapData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.io))
    val seq = new Sequential()
    seq.add(layer1)
    seq.add(layer2)
    seq.add(layer3)
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(3, 4), Memory.Format.nc)))
    val input1 = Tensor[Float](3, 4).rand()
    val input2 = Tensor[Float](3, 4).rand()
    val output1 = seq.forward(input1)
    output1 should be(input1)
    val output2 = seq.forward(input2)
    output2 should be(input2)

    val gradOutput1 = Tensor[Float](3, 4).rand()
    val gradInput1 = seq.backward(input1, gradOutput1)
    gradInput1 should be(gradOutput1)

    val gradOutput2 = Tensor[Float](3, 4).rand()
    val gradInput2 = seq.backward(input2, gradOutput2)
    gradInput2 should be(gradOutput2)
  }

  "Sequential" should "be correct when auto add memory reorder" in {
    val layer1 = ReorderMemory.create(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory.create(
      NativeData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.io),
      NativeData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.io))
    val layer3 = ReorderMemory.create(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    val seq = Sequential()
    seq.add(layer1)
    seq.add(layer2)
    seq.add(layer3)
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(3, 4), Memory.Format.nc)))
    val input1 = Tensor[Float](3, 4).rand()
    val input2 = Tensor[Float](3, 4).rand()
    println(s"Input1 is $input1")
    println(s"Input2 is $input2")
    val output1 = seq.forward(input1)
    output1 should be(input1)
    val output2 = seq.forward(input2)
    output2 should be(input2)

    val gradOutput1 = Tensor[Float](3, 4).rand()
    val gradInput1 = seq.backward(input1, gradOutput1)
    gradInput1 should be(gradOutput1)

    val gradOutput2 = Tensor[Float](3, 4).rand()
    val gradInput2 = seq.backward(input2, gradOutput2)
    gradInput2 should be(gradOutput2)
  }

  "seq with java serialization" should "work correctly" in {
    val layer1 = ReorderMemory.create(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory.create(
      NativeData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.io),
      NativeData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.io))
    val layer3 = ReorderMemory.create(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    val seq = Sequential()
    seq.add(layer1)
    seq.add(layer2)
    seq.add(layer3)
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(3, 4), Memory.Format.nc)))

    val input = Tensor[Float](3, 4).rand()
    val gradOutput = Tensor[Float](3, 4).rand()

    seq.forward(input)
    seq.backward(input, gradOutput)

    val cloned = SerializationUtils.clone(seq)
    cloned.compile(Phase.TrainingPhase, Array(HeapData(Array(3, 4), Memory.Format.nc)))

    cloned.forward(input)
    cloned.backward(input, gradOutput)

    Tools.dense(seq.output) should be (Tools.dense(cloned.output))
    Tools.dense(seq.gradInput) should be (Tools.dense(cloned.gradInput))
  }

  "compile with input" should "work correctly" in {
    val inputShape = Array(4, 1, 28, 28)
    val outputShape = Array(4, 10)

    val model = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(SpatialConvolution(1, 20, 5, 5).setName("conv1"))
      .add(MaxPooling(2, 2, 2, 2).setName("pool1"))
      .add(SpatialConvolution(20, 50, 5, 5).setName("conv2"))
      .add(MaxPooling(2, 2, 2, 2).setName("pool2"))
      .add(Linear(50 * 4 * 4, 500).setName("ip1"))
      .add(ReLU().setName("relu1"))
      .add(Linear(500, 10).setName("ip2"))
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nc)))

    model.compile(TrainingPhase)

    val input = Tensor[Float](inputShape).rand(-1, 1)
    val gradOutput = Tensor[Float](outputShape).rand(-1, 1)

    model.forward(input)
    model.backward(input, gradOutput)
  }

  "no input" should "throw exception" in {
    val inputShape = Array(4, 1, 2, 2)
    val outputShape = Array(4, 1, 2, 2)

    val model1 = Sequential()
      .add(ReLU().setName("relu1"))
      .add(ReorderMemory(HeapData(outputShape, Memory.Format.nc)))

    val model2 = Sequential()
      .add(Sequential()
        .add(ReLU().setName("relu1"))
        .add(ReorderMemory(HeapData(outputShape, Memory.Format.nc))))

    val model3 = Sequential()
        .add(ConcatTable().add(ReLU()).add(ReLU()))

    List(model1, model2, model3).foreach { model =>
      intercept[IllegalArgumentException] {
        model.compile(TrainingPhase)
      }
    }
  }
}
