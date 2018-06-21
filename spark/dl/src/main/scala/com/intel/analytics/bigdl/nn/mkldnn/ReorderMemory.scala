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

class ReorderMemory(inputFormat: MemoryData, outputFormat: MemoryData,
  gradInputFormat: MemoryData, gradOutputFormat: MemoryData) extends MklDnnLayer {

  _inputFormats = Array(inputFormat)
  _gradInputFormats = Array(gradInputFormat)
  _outputFormats = Array(outputFormat)
  _gradOutputFormats = Array(gradOutputFormat)
  _gradOutputFormatsForWeight = Array(gradOutputFormat)

  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]): Array[Array[Int]] = {
    Array(outputFormat.shape)
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    this.runtime = runtime
    val inputMemDesc = MklDnn.MemoryDescInit(inputFormat.shape.length, inputFormat.shape,
      DataType.F32, inputFormat.layout)
    val inputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(inputMemDesc, runtime.engine)
    val inputPrimitives = Array(MklDnn.PrimitiveCreate0(inputPrimDesc))

    val outputMemDesc = MklDnn.MemoryDescInit(outputFormat.shape.length, outputFormat.shape,
      DataType.F32, outputFormat.layout)
    val outputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(outputMemDesc, runtime.engine)
    val outputPrimitives = Array(MklDnn.PrimitiveCreate0(outputPrimDesc))

    val fwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(inputPrimDesc, outputPrimDesc)
    val fwdReorderPrim = MklDnnOps.primitiveCreate2(fwdReorderPrimDesc, inputPrimitives,
      Array(0), 1, outputPrimitives, 1)

    fwdMemPrims = inputPrimitives ++ outputPrimitives
    updateOutputPrimitives = Array(fwdReorderPrim)
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    this.runtime = runtime
    val gradInputMemDesc = MklDnn.MemoryDescInit(gradInputFormat.shape.length,
      gradInputFormat.shape, DataType.F32, gradInputFormat.layout)
    val gradInputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(gradInputMemDesc, runtime.engine)
    val gradInputPrimitives = Array(MklDnn.PrimitiveCreate0(gradInputPrimDesc))

    val gradOutputMemDesc = MklDnn.MemoryDescInit(gradOutputFormat.shape.length,
      gradOutputFormat.shape, DataType.F32, gradOutputFormat.layout)
    val gradOutputPrimDesc = MklDnn.MemoryPrimitiveDescCreate(gradOutputMemDesc, runtime.engine)
    val gradOutputPrimitives = Array(MklDnn.PrimitiveCreate0(gradOutputPrimDesc))

    val bwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(gradOutputPrimDesc,
      gradInputPrimDesc)
    val bwdReorderPrim = MklDnnOps.primitiveCreate2(bwdReorderPrimDesc, gradOutputPrimitives,
      Array(0), 1, gradInputPrimitives, 1)

    bwdMemPrims = gradOutputPrimitives ++ gradInputPrimitives
    updateGradInputPrimitives = Array(bwdReorderPrim)
  }

  override def toString(): String = {
    s"nn.mkl.ReorderMemory(${inputFormat} -> ${outputFormat})"
  }
}

object ReorderMemory {
  def apply(inputFormat: MemoryData, outputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat,
      inputFormat.cloneFormat(), outputFormat.cloneFormat())
  }

  def apply(inputFormat: MemoryData, outputFormat: MemoryData, gradInputFormat: MemoryData,
    gradOutputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat, gradInputFormat, gradOutputFormat)
  }
}
