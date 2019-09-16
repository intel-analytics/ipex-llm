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
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}

class JoinTableSpec extends BigDLSpecHelper {
  "Join table" should "work correctly" in {
    val layer = JoinTable(1)
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
    model.add(ReorderMemory.create(NativeData(Array(4, 2), Memory.Format.nc),
      HeapData(Array(4, 2), Memory.Format.nc), NativeData(Array(4, 2), Memory.Format.nc),
      HeapData(Array(4, 2), Memory.Format.nc)))
    model.compile(Phase.TrainingPhase, Array(HeapData(Array(2, 2), Memory.Format.nc)))
    model.forward(Tensor[Float](T(T(1, 2), T(3, 4)))) should be(Tensor[Float](T(
      T(1, 2),
      T(3, 4),
      T(1, 2),
      T(3, 4)
    )))
    val dnnGrad = model.backward(Tensor[Float](T(T(1, 2), T(3, 4))), T(
      Tensor[Float](T(
        T(4, 5),
        T(6, 7),
        T(1, 3),
        T(4, 2)
      ))
    )).asInstanceOf[Tensor[Float]]
    val heapGrad = Tensor[Float](2, 2)
    heapGrad.copy(dnnGrad)
    heapGrad should be(
      Tensor[Float](T(T(5, 8), T(10, 9)))
    )
  }

  "int8 of join table" should "work correctly" in {
    val inputShape1 = Array[Int](4, 2, 4, 4)
    val inputShape2 = Array[Int](4, 4, 4, 4)

    val input1 = Input(inputShape1, Memory.Format.nchw).inputs()
    val input2 = Input(inputShape2, Memory.Format.nchw).inputs()
    val conv1 = SpatialConvolution(2, 4, 5, 5, 1, 1, 2, 2).inputs(input1)
    val conv2 = SpatialConvolution(4, 4, 1, 1, 1, 1, 0, 0).inputs(input2)
    val joinTable = JoinTable(2).inputs(conv1, conv2)
    val output = Output(Memory.Format.nchw).inputs(joinTable)

    val model = DnnGraph(Seq(input1, input2), Seq(output))
    model.evaluate()

    val tensor1 = Tensor[Float](inputShape1).rand(-1, 1)
    val tensor2 = Tensor[Float](inputShape2).rand(-0.1, 0.1)
    val tableInput = T(tensor1, tensor2)

    model.setWeightDimMask(1, overrideSubmodules = true)
    model.compile(InferencePhase)
    model.forward(tableInput)

    model.calcScales(tableInput)

    val outputOfModel = Tensor[Float]()
      .resizeAs(model.output.toTensor[Float])
      .copy(model.output.toTensor[Float])


    model.clearState()

    val quant = model.quantize().asInstanceOf[DnnGraph]
    quant.compile(InferencePhase)
    quant.forward(tableInput)

    Equivalent.nearequals(outputOfModel, quant.output.toTensor[Float], 1e-1) should be (true)

    model.release()
    quant.release()
  }
}
