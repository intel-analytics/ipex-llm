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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

class ReorderMemory(inputFormat: MemoryData, outputFormat: MemoryData)
  extends TensorModule[Float] with MklDnnModule {

  private var _inputFormat = inputFormat
  private var _outputFormat = outputFormat

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    MklDnnOps.streamSubmit(
      runtime.stream, 1, forwardPrimitives, 1, Array(inputPrimitives(0), outputPrimitives(0)),
      Array(input, output)
    )
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    MklDnnOps.streamSubmit(runtime.stream, 1, backwardPrimitives, 1,
      Array(outputPrimitives(0), inputPrimitives(0)), Array(gradOutput, gradInput))
    gradInput
  }

  override private[mkldnn] def inferOutputFormats(): Array[MemoryData] = {
    _outputFormat.setShape(inputFormat.shape)
    _outputFormat.setLayout(inputFormat.layout)
    Array(_outputFormat)
  }

  override private[mkldnn] def setInputFormats(formats: Array[MemoryData]): Unit = {
    require(formats.length == 1, "input tensor number should be one")
    require(formats(0).shape != null, "input tensor shape should be defined")
    require(formats(0).layout != MklDnn.MemoryFormat.any, "input tensor shape should be defined")
    require(MemoryData.isCompatible(formats, Array(inputFormat)), "memory format is not compatible")
    _inputFormat.setShape(formats(0).shape)
    _inputFormat.setLayout(formats(0).layout)
  }

  /**
   * Init the MKL-DNN primitives for the model
   * @param runtime
   */
  override private[mkldnn] def initPrimitives(runtime: MklDnnRuntime) = {
    this.runtime = runtime
    val inputMemDesc = MklDnn.MemoryDescInit(_inputFormat.shape.length, _inputFormat.shape,
      MklDnn.DataType.f32, _inputFormat.layout)
    val inputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(inputMemDesc, runtime.engine)
    inputPrimitives = Array(MklDnn.PrimitiveCreate0(inputPrimDesc))

    val outputMemDesc = MklDnn.MemoryDescInit(_outputFormat.shape.length, _outputFormat.shape,
      MklDnn.DataType.f32, _outputFormat.layout)
    val outputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(outputMemDesc, runtime.engine)
    outputPrimitives = Array(MklDnn.PrimitiveCreate0(outputPrimDesc))

    val fwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(inputPrimDesc, outputPrimDesc)
    val fwdReorderPrim = MklDnnOps.primitiveCreate2(fwdReorderPrimDesc, inputPrimitives,
      Array(0), 1, outputPrimitives, 1)

    forwardPrimitives = Array(fwdReorderPrim)

    val bwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(outputPrimDesc, inputPrimDesc)
    val bwdReorderPrim = MklDnnOps.primitiveCreate2(bwdReorderPrimDesc, outputPrimitives,
      Array(0), 1, inputPrimitives, 1)

    backwardPrimitives = Array(bwdReorderPrim)
  }

  override private[mkldnn] def allocateMemory() = {
    _inputFormat match {
      case d: NativeData =>
        gradInput = DnnTensor[Float](d.shape)
      case d: HeapData =>
        gradInput = Tensor[Float](d.shape)
      case _ => throw new UnsupportedOperationException("memory format is not supported")
    }
    _outputFormat match {
      case d: NativeData =>
        output = DnnTensor[Float](d.shape)
      case d: HeapData =>
        output = Tensor[Float](d.shape)
      case _ => throw new UnsupportedOperationException("memory format is not supported")
    }
  }

  override private[mkldnn] def expectInputFormats = {
    Array(_outputFormat)
  }
}

object ReorderMemory {
  def apply(inputFormat: MemoryData, outputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat)
  }
}
