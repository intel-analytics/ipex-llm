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
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

class ReorderMemory(inputFormat: MemoryData, outputFormat: MemoryData,
  gradInputFormat: MemoryData, gradOutputFormat: MemoryData
) extends MklDnnLayer {

  _outputFormats = Array(outputFormat)
  _gradInputFormats = Array(gradInputFormat)

  private var realInput : Array[MemoryData] = null
  private var realOutput : Array[MemoryData] = null
  private var realgradInput : Array[MemoryData] = null
  private var realgradOutput : Array[MemoryData] = null

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = if (inputFormat == null) inputs else Array(inputFormat)
    require(_inputFormats.length == 1, "Only accept one tensor as input")

    if (outputFormat == null) _outputFormats = _inputFormats
    require(_inputFormats(0).shape.product == _outputFormats(0).shape.product,
      "input output memory not match")

    val inputShape = _inputFormats(0).shape
    val outputShape = _outputFormats(0).shape
    val inputLayout = _inputFormats(0).layout
    val outputLayout = _outputFormats(0).layout
    realInput = _inputFormats
    realOutput = _outputFormats

    if (inputLayout != outputLayout) {
      if (inputLayout == Memory.Format.nhwc) {
        // remind: if format of input MemoryData is nhwc, its shape should be output shape
        realInput = Array(HeapData(outputShape, inputLayout))
      } else if (outputLayout == Memory.Format.nhwc) {
        // remind: if format of output MemoryData is nhwc, its shape should be input shape
       realOutput = Array(HeapData(inputShape, outputLayout))
      }
    }

    val fwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(
      realInput(0).getPrimitiveDescription(runtime),
      realOutput(0).getPrimitiveDescription(runtime))
    val fwdReorderPrim = MklDnn.PrimitiveCreate2(fwdReorderPrimDesc,
      Array(realInput(0).getPrimitive(runtime)), Array(0), 1,
      Array(realOutput(0).getPrimitive(runtime)), 1)

    updateOutputPrimitives = Array(fwdReorderPrim)

    // recover to original data
    output = initTensor(realOutput(0))
    (_inputFormats, _outputFormats)
  }

  override def getUpdateGradInputMemoryPrimitives(): Array[Long] = {
    realgradOutput.map(_.getPrimitive(runtime)) ++ realgradInput.map(_.getPrimitive(runtime))
  }

  override def getUpdateOutputMemoryPrimitives(): Array[Long] = {
    realInput.map(_.getPrimitive(runtime)) ++ realOutput.map(_.getPrimitive(runtime))
  }
  override def updateOutput(input: Activity): Activity = {
    if (_inputFormats(0).layout == _outputFormats(0).layout) {
      output = input
    } else {
      output = super.updateOutput(input)
      output.toTensor[Float].resize(_outputFormats(0).shape)
    }
    output
  }

  override private[bigdl] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    _gradInputFormats = (gradInputFormat, inputFormat) match {
      case (null, null) => inputFormats()
      case (null, x) => Array(x)
      case (x, _) => Array(x)
    }

    _gradOutputFormats = if (gradOutputFormat == null) grads else Array(gradOutputFormat)
    require(_gradOutputFormats.length == 1, "Only accept one tensor as input")
    require(_gradOutputFormats(0).shape.product == _gradInputFormats(0).shape.product,
      "input output memory not match")

    val gradInputShape = _gradInputFormats(0).shape
    val gradOutputShape = _gradOutputFormats(0).shape
    val gradInputLayout = _gradInputFormats(0).layout
    val gradOutputLayout = _gradOutputFormats(0).layout
    realgradInput = _gradInputFormats
    realgradOutput = _gradOutputFormats

    if (gradInputLayout != gradOutputLayout) {
      if (gradOutputLayout == Memory.Format.nhwc) {
        // todo: if format of gradOutput MemoryData is nhwc, its shape should be gradInput shape
        realgradOutput = Array(HeapData(gradInputShape, gradOutputLayout))
      } else if (gradInputLayout == Memory.Format.nhwc) {
        // todo: if format of gradInput MemoryData is nhwc, its shape should be gradOutput shape
        realgradInput = Array(HeapData(gradOutputShape, gradInputLayout))
      }
    }

    val bwdReorderPrimDesc = MklDnn.ReorderPrimitiveDescCreate(
      realgradOutput(0).getPrimitiveDescription(runtime),
      realgradInput(0).getPrimitiveDescription(runtime))
    val bwdReorderPrim = MklDnn.PrimitiveCreate2(bwdReorderPrimDesc,
      realgradOutput.map(_.getPrimitive(runtime)), Array(0), 1,
      realgradInput.map(_.getPrimitive(runtime)), 1)

    updateGradInputPrimitives = Array(bwdReorderPrim)
    gradInput = initTensor(realgradInput(0))
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (_gradInputFormats(0).layout == _gradOutputFormats(0).layout) {
      gradInput = gradOutput
    } else {
      gradInput = super.updateGradInput(input, gradOutput)
      gradInput.toTensor[Float].resize(_gradInputFormats(0).shape)
    }
    gradInput
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
  def apply(inputFormat: MemoryData, outputFormat: MemoryData, gradInputFormat: MemoryData,
    gradOutputFomat: MemoryData): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat, gradInputFormat, gradOutputFomat)
  }

  def apply(outputFormat: MemoryData, gradInputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(null, outputFormat, gradInputFormat, null)
  }

  def apply(outputFormat: MemoryData): ReorderMemory = {
    new ReorderMemory(null, outputFormat, null, null)
  }
}
