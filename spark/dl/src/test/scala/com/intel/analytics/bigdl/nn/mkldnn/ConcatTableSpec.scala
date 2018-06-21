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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}

class ConcatTableSpec extends BigDLSpecHelper {
  "ConcatTable" should "throw exception when input layout is different" in {
    val container = ConcatTable()
    container.add(Input(Array(1, 2, 3, 4), Memory.Format.nchw))
    container.add(Input(Array(1, 2, 3, 4), Memory.Format.nhwc))

    intercept[IllegalArgumentException] {
      container.compile(Phase.TrainingPhase, Array(HeapData(Array(1, 2, 3, 4), Memory.Format.nchw)))
    }
  }

  "ConcatTable" should "throw exception when input format is different" in {
    val container = ConcatTable()
    container.add(ReorderMemory(HeapData(Array(1, 2, 3, 4), Memory.Format.nchw),
      HeapData(Array(1, 2, 3, 4), Memory.Format.nchw),
      HeapData(Array(1, 2, 3, 4), Memory.Format.nchw),
      HeapData(Array(1, 2, 3, 4), Memory.Format.nchw)))
    container.add(ReorderMemory(NativeData(Array(1, 2, 3, 4), Memory.Format.nchw),
      HeapData(Array(1, 2, 3, 4), Memory.Format.nchw),
      NativeData(Array(1, 2, 3, 4), Memory.Format.nchw),
      HeapData(Array(1, 2, 3, 4), Memory.Format.nchw)))

    intercept[IllegalArgumentException] {
      container.compile(Phase.TrainingPhase, Array(HeapData(Array(1, 2, 3, 4), Memory.Format.nchw)))
    }
  }

  "ConcatTable" should "throw exception when input shape is different" in {
    val container = ConcatTable()
    container.add(Input(Array(1, 2, 3, 4), Memory.Format.nchw))
    container.add(Input(Array(1, 3, 4, 2), Memory.Format.nchw))

    intercept[IllegalArgumentException] {
      container.compile(Phase.TrainingPhase, Array(HeapData(Array(1, 2, 3, 4), Memory.Format.nchw)))
    }
  }

  "ConcatTable" should "be good" in {
    val container = ConcatTable()
    container.add(ReorderMemory(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc)))
    val subcontainer = Sequential()
    subcontainer.add(ReorderMemory(
      HeapData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.nc)))
    subcontainer.add(ReorderMemory(NativeData(Array(3, 4), Memory.Format.io),
      NativeData(Array(3, 4), Memory.Format.nc)))
    subcontainer.add(ReorderMemory(HeapData(Array(3, 4), Memory.Format.nc),
      NativeData(Array(3, 4), Memory.Format.io)))
    container.add(subcontainer)

    container.compile(Phase.TrainingPhase, Array(HeapData(Array(3, 4), Memory.Format.nc)))
    val input1 = Tensor[Float](3, 4).rand()
    val output1 = container.forward(input1).toTable
    output1(1).asInstanceOf[Tensor[Float]] should be(input1)
    output1(2).asInstanceOf[Tensor[Float]] should be(input1)

    val grad1 = Tensor[Float](3, 4).rand()
    container.backward(input1, T(grad1, grad1)) should be(grad1 * 2)
    
    val input2 = Tensor[Float](3, 4).rand()
    val output2 = container.forward(input2).toTable
    output2(1).asInstanceOf[Tensor[Float]] should be(input2)
    output2(2).asInstanceOf[Tensor[Float]] should be(input2)

    val grad2 = Tensor[Float](3, 4).rand()
    container.backward(input1, T(grad2, grad2)) should be(grad2 * 2)
  }
}
