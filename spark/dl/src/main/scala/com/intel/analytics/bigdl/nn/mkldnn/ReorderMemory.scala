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

class ReorderMemory(outputFormat: MemoryData, gradInputFormat: MemoryData) extends MklDnnLayer {

  _outputFormats = Array(outputFormat)
  _gradInputFormats = Array(gradInputFormat)


  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length == 1, "Only accept one tensor as input")
    _inputFormats = inputs

    val fwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(
      inputs(0).getPrimitiveDescription(runtime), outputFormat.getPrimitiveDescription(runtime))
    val fwdReorderPrim = MklDnnOps.primitiveCreate2(fwdReorderPrimDesc,
      Array(inputs(0).getPrimitive(runtime)), Array(0), 1,
      Array(outputFormat.getPrimitive(runtime)), 1)

    updateOutputPrimitives = Array(fwdReorderPrim)
    output = initTensor(outputFormat)
    (inputs, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == 1, "Only accept one tensor as input")
    _gradOutputFormats = grads
    _gradOutputFormatsForWeight = grads

    val bwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(
      grads(0).getPrimitiveDescription(runtime), gradInputFormat.getPrimitiveDescription(runtime))
    val bwdReorderPrim = MklDnnOps.primitiveCreate2(bwdReorderPrimDesc,
      grads.map(_.getPrimitive(runtime)), Array(0), 1,
      _gradInputFormats.map(_.getPrimitive(runtime)), 1)

    updateGradInputPrimitives = Array(bwdReorderPrim)
    gradInput = initTensor(gradInputFormat)
    (grads, _gradInputFormats)
  }

  override def toString(): String = {
    if (_inputFormats != null) {
      s"nn.mkl.ReorderMemory(${_inputFormats(0)} -> ${outputFormat})"
    } else {
      s"nn.mkl.ReorderMemory(_ -> ${outputFormat})"
    }
  }
}

object ReorderMemory {
  def apply(outputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(outputFormat, outputFormat.cloneFormat())
  }

  def apply(outputFormat: MemoryData, gradInputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(outputFormat, gradInputFormat)
  }
}
