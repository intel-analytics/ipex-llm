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

import com.intel.analytics.bigdl.mkl.{DataType, Memory}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}
import org.apache.commons.lang3.SerializationUtils

class CAddTableSpec extends BigDLSpecHelper {
  "CAddTable" should "be correct" in {
    val layer = CAddTable()
    val model = Sequential()
    val concat = ConcatTable()
    concat.add(ReorderMemory.create(HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc), HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc)))
    concat.add(ReorderMemory.create(HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc), HeapData(Array(2, 2), Memory.Format.nc),
      NativeData(Array(2, 2), Memory.Format.nc)))
    model.add(concat)
    model.add(layer)
    model.add(ReorderMemory.create(NativeData(Array(2, 2), Memory.Format.nc),
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
    implicit object Owner extends MemoryOwner {
    }
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
    Owner.releaseResources()
  }

  "CAddTable u8" should "be correct" in {
    val shape = Array(4, 3, 5, 5)
    val model = Sequential()
    val concat = ConcatTable()
    val cadd = CAddTable()

    model.add(Input(shape, Memory.Format.nchw))
    model.add(concat).add(cadd)

    val input = Tensor[Float](shape).rand(0, 1)

    val nativeData1 = NativeData(shape, Memory.Format.nhwc, DataType.U8)
    val nativeData2 = NativeData(shape, Memory.Format.nhwc, DataType.U8)

    nativeData1.setMask(0)
    nativeData1.setScales(Array(255.0f / input.clone().abs().max()))

    nativeData2.setMask(0)
    nativeData2.setScales(Array(255.0f / input.clone().abs().max()))

    concat.add(ReorderMemory(nativeData1))
    concat.add(ReorderMemory(nativeData2))

    model.add(ReorderMemory(HeapData(shape, Memory.Format.nchw)))

    model.evaluate()
    model.compile(InferencePhase)
    model.forward(input)


    val seq2 = Sequential()
      .add(Input(shape, Memory.Format.nchw))
      .add(ConcatTable()
        .add(ReorderMemory(NativeData(shape, Memory.Format.nhwc)))
        .add(ReorderMemory(NativeData(shape, Memory.Format.nchw))))
      .add(CAddTable())
      .add(ReorderMemory(HeapData(shape, Memory.Format.nchw)))

    seq2.evaluate()
    seq2.compile(InferencePhase)
    seq2.forward(input)

    println()
  }
}
