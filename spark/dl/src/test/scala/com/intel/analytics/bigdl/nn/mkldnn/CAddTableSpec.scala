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
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}
import org.apache.commons.lang3.SerializationUtils

class CAddTableSpec extends BigDLSpecHelper {
  "CAddTable" should "be correct" in {
    val layer = CAddTable()
    val model = Sequential()
    val concat = ConcatTable()
    concat.add(ReorderMemory(HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc), HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc)))
    concat.add(ReorderMemory(HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc), HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc)))
    model.add(concat)
    model.add(layer)
    model.add(ReorderMemory(NativeData(Array(2, 2), Memory.Format.nc),
      HeapData(Array(2, 2), Memory.Format.nc), NativeData(Array(2, 2), Memory.Format.nc),
      HeapData(Array(2, 2), Memory.Format.nc)))
    model.compile(Phase.TrainingPhase, Array(HeapData(Array(2, 2), Memory.Format.nc)))
    model.forward(Tensor[Float](T(T(1, 2), T(3, 4)))) should be(Tensor[Float](T(
      T(2, 4),
      T(6, 8)
    )))
    val dnnGrad = model.backward(Tensor[Float](T(T(1, 2), T(3, 4))), T(
      Tensor[Float](T(
        T(4, 5),
        T(6, 7)
      ))
    )).asInstanceOf[Tensor[Float]]
    val heapGrad = Tensor[Float](2, 2)
    heapGrad.copy(dnnGrad)
    heapGrad should be (
      Tensor[Float](T(T(8, 10), T(12, 14)))
    )
  }

  "caddtable with java serialization" should "work correctly" in {
    val shape = Array(2, 3, 4, 4)
    val _1 = Tensor(shape).rand(-1, 1)
    val _2 = Tensor(shape).rand(-1, 1)

    val input1 = DnnTensor(shape).copy(_1)
    val input2 = DnnTensor(shape).copy(_2)

    val cat = CAddTable()
    cat.setRuntime(new MklDnnRuntime)
    cat.initFwdPrimitives(Array(
      HeapData(shape, Memory.Format.nchw),
      HeapData(shape, Memory.Format.nchw)), TrainingPhase)
    cat.initBwdPrimitives(Array(
      HeapData(shape, Memory.Format.nchw),
      HeapData(shape, Memory.Format.nchw)), TrainingPhase)

    cat.forward(T(input1, input2))

    val cloned = SerializationUtils.clone(cat)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(
      HeapData(shape, Memory.Format.nchw),
      HeapData(shape, Memory.Format.nchw)), TrainingPhase)
    cloned.initBwdPrimitives(Array(
      HeapData(shape, Memory.Format.nchw),
      HeapData(shape, Memory.Format.nchw)), TrainingPhase)
    cloned.forward(T(input1, input2))

    Tools.dense(cat.output) should be (Tools.dense(cloned.output))

    val gradOutput = Tensor(shape).rand(-1, 1)
    cat.backward(T(input1, input2), gradOutput)
    cloned.backward(T(input1, input2), gradOutput)

    Tools.dense(cat.gradInput.toTable(1)) should be (Tools.dense(cloned.gradInput.toTable(1)))
    Tools.dense(cat.gradInput.toTable(2)) should be (Tools.dense(cloned.gradInput.toTable(2)))
  }
}
