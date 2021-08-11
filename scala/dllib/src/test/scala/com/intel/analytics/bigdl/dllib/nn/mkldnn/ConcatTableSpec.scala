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

class ConcatTableSpec extends BigDLSpecHelper {
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
    container.add(ReorderMemory.create(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc)))
    val subcontainer = Sequential()
    subcontainer.add(ReorderMemory.create(
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
    val nativeGrad = container.backward(input1, T(grad1, grad1)).asInstanceOf[Tensor[Float]]
    val heapGrad = Tensor[Float](3, 4).copy(nativeGrad)
    heapGrad should be(grad1 * 2)
    val input2 = Tensor[Float](3, 4).rand()
    val output2 = container.forward(input2).toTable
    output2(1).asInstanceOf[Tensor[Float]] should be(input2)
    output2(2).asInstanceOf[Tensor[Float]] should be(input2)

    val grad2 = Tensor[Float](3, 4).rand()
    val nativeGrad2 = container.backward(input1, T(grad2, grad2)).asInstanceOf[Tensor[Float]]
    val heapGrad2 = Tensor[Float](3, 4).copy(nativeGrad2)
    heapGrad2 should be(grad2 * 2)
  }

  "concat table with java serialization" should "work correctly" in {
    val shape = Array(2, 2)
    val input = Tensor(shape).fill(1)
    val gradOutput = T(Tensor(shape).fill(2), Tensor(shape).fill(2))

    val ct = ConcatTable()
    ct.add(Identity())
    ct.add(Identity())

    ct.compile(TrainingPhase, Array(HeapData(shape, Memory.Format.nc)))

    val cloned = SerializationUtils.clone(ct)
    cloned.compile(TrainingPhase, Array(HeapData(shape, Memory.Format.nc)))

    ct.forward(input)
    ct.backward(input, gradOutput)

    cloned.forward(input)
    cloned.backward(input, gradOutput)

    Tools.dense(ct.output.toTable(1)) should be(Tools.dense(cloned.output.toTable(1)))
    Tools.dense(ct.output.toTable(2)) should be(Tools.dense(cloned.output.toTable(2)))

    Tools.dense(ct.gradInput) should be(Tools.dense(cloned.gradInput))
  }

  "ConcatTable with U8" should "be good" in {
    val shape = Array(4, 3, 5, 5)
    val input = Tensor[Float](shape).rand(0, 1)

    println(input)

    val nativeData1 = NativeData(shape, Memory.Format.nchw, DataType.U8)
    val nativeData2 = NativeData(shape, Memory.Format.nchw, DataType.U8)
    val heapData = HeapData(shape, Memory.Format.nchw, DataType.F32)
    heapData.setMask(0)
    val inputScales = Array(input.clone().abs().max())
    heapData.setScales(inputScales.map(x => 255f / x))

    val concat = ConcatTable()

    concat.add(ReorderMemory(nativeData1))
    concat.add(ReorderMemory(nativeData2))
    concat.compile(Phase.InferencePhase, Array(heapData))

    concat.forward(input)

    val output1 = new Array[Byte](shape.product)
    Memory.CopyPtr2ByteArray(
      concat.output.toTable[Tensor[Float]](1)
        .asInstanceOf[DnnTensor[Byte]].storageAddress(),
      0, output1, 0, shape.product, 1)
    output1.foreach(println)

    val output2 = new Array[Byte](shape.product)
    Memory.CopyPtr2ByteArray(
      concat.output.toTable[Tensor[Float]](1)
        .asInstanceOf[DnnTensor[Byte]].storageAddress(),
      0, output2, 0, shape.product, 1)
    output2.foreach(println)

    output1 should be (output2)
  }
}
