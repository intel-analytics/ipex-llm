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

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn}
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

class ReorderMemory(inputFormat: MemoryData, outputFormat: MemoryData)
  extends TensorModule[Float] with MklDnnLayer {

  private var _inputFormat = inputFormat
  private var _outputFormat = outputFormat

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    MklDnnOps.streamSubmit(
      runtime.stream, 1, updateOutputPrimitives, 1, Array(inputPrimitives(0), outputPrimitives(0)),
      Array(input, output)
    )
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives, 1,
      Array(gradOutputPrimitives(0), gradInputPrimitives(0)), Array(gradOutput, gradInput))
    gradInput
  }

  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]): Array[Array[Int]] = {
    Array(outputFormat.shape)
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    this.runtime = runtime
    val inputMemDesc = MklDnn.MemoryDescInit(_inputFormat.shape.length, _inputFormat.shape,
      DataType.F32, _inputFormat.layout)
    val inputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(inputMemDesc, runtime.engine)
    inputPrimitives = Array(MklDnn.PrimitiveCreate0(inputPrimDesc))

    val outputMemDesc = MklDnn.MemoryDescInit(_outputFormat.shape.length, _outputFormat.shape,
      DataType.F32, _outputFormat.layout)
    val outputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(outputMemDesc, runtime.engine)
    outputPrimitives = Array(MklDnn.PrimitiveCreate0(outputPrimDesc))

    val fwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(inputPrimDesc, outputPrimDesc)
    val fwdReorderPrim = MklDnnOps.primitiveCreate2(fwdReorderPrimDesc, inputPrimitives,
      Array(0), 1, outputPrimitives, 1)

    updateOutputPrimitives = Array(fwdReorderPrim)
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    this.runtime = runtime
    val gradInputMemDesc = MklDnn.MemoryDescInit(_inputFormat.shape.length, _inputFormat.shape,
      DataType.F32, _inputFormat.layout)
    val gradInputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(gradInputMemDesc, runtime.engine)
    gradInputPrimitives = Array(MklDnn.PrimitiveCreate0(gradInputPrimDesc))

    val gradOutputMemDesc = MklDnn.MemoryDescInit(_outputFormat.shape.length, _outputFormat.shape,
      DataType.F32, _outputFormat.layout)
    val gradOutputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(gradOutputMemDesc, runtime.engine)
    gradOutputPrimitives = Array(MklDnn.PrimitiveCreate0(gradOutputPrimDesc))

    val bwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(gradOutputPrimDesc,
      gradInputPrimDesc)
    val bwdReorderPrim = MklDnnOps.primitiveCreate2(bwdReorderPrimDesc, gradOutputPrimitives,
      Array(0), 1, gradInputPrimitives, 1)

    updateGradInputPrimitives = Array(bwdReorderPrim)
  }

  override private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase) = {}

  override private[mkldnn] def initMemory() = {
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

  override private[mkldnn] def inputFormats() = Array(inputFormat)

  override private[mkldnn] def gradInputFormats() = Array(inputFormat)

  override private[mkldnn] def outputFormats() = Array(outputFormat)

  override private[mkldnn] def gradOutputFormats() = (Array(outputFormat), Array(outputFormat))

  override def toString(): String = {
    s"nn.mkl.ReorderMemory(${inputFormat} -> ${outputFormat})"
  }
}

object ReorderMemory {
  def apply(inputFormat: MemoryData, outputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat)
  }
}
