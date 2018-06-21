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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper

class SequentialSpec extends BigDLSpecHelper {
  "Sequential" should "not be called add after compilation" in {
    val layer = ReorderMemory(NativeData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory(NativeData(Array(3, 4), Memory.Format.nc))
    val seq = new Sequential()
    seq.add(layer)
    seq.compile(Phase.TrainingPhase)
    intercept[IllegalArgumentException] {
      seq.add(layer2)
    }
  }

  "Sequential" should "be correct when no memory reorder happened" in {
    val layer1 = ReorderMemory(new HeapData(Array(3, 4), Memory.Format.nc),
      new NativeData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory(new NativeData(Array(3, 4), Memory.Format.nc),
      new NativeData(Array(3, 4), Memory.Format.io))
    val layer3 = ReorderMemory(new NativeData(Array(3, 4), Memory.Format.io),
      new HeapData(Array(3, 4), Memory.Format.nc))
    val seq = new Sequential()
    seq.add(layer1)
    seq.add(layer2)
    seq.add(layer3)
    seq.compile(Phase.TrainingPhase)
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
    val layer1 = ReorderMemory(new HeapData(Array(3, 4), Memory.Format.nc),
      new HeapData(Array(3, 4), Memory.Format.nc))
    val layer2 = ReorderMemory(new NativeData(Array(3, 4), Memory.Format.nc),
      new NativeData(Array(3, 4), Memory.Format.io))
    val layer3 = ReorderMemory(new HeapData(Array(3, 4), Memory.Format.nc),
      new HeapData(Array(3, 4), Memory.Format.nc))
    val seq = new Sequential()
    seq.add(layer1)
    seq.add(layer2)
    seq.add(layer3)
    seq.compile(Phase.TrainingPhase)
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
}
