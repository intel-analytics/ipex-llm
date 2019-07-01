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

  private def initMemory(src: MemoryData, shape: Array[Int], layout: Int)
    : Array[MemoryData] = {
    val ret = src match {
      case h: HeapData => Array(HeapData(shape, layout, src.dataType))
      case n: NativeData => Array(NativeData(shape, layout, src.dataType))
      case _ => throw new UnsupportedOperationException("Not support such memory format")
    }

    ret(0).setMask(src.mask)
    ret(0).setScales(src.scales)
    ret.asInstanceOf[Array[MemoryData]]
  }

  private def shapeToString(shape: Array[Int]): String = {
    var name = ""
    shape.foreach(s => name += s.toString + ",")
    name
  }

  private def reshapeOutputIfNeeded(format: MemoryData, tensor: Tensor[Float]): Unit = {
    // must pay attention to the shape of tensor when format is nhwc,
    // the Tensor's shape in BigDL always be relevant with the format, such as
    // [4, 3, 224, 224] will be nchw and [4, 224, 224, 3] will be nhwc.
    // but for mkldnn, it always uses the nchw format shape, library will use
    // correct shape by the format.
    if (format.layout == Memory.Format.nhwc && format.isInstanceOf[HeapData]) {
      tensor.toTensor[Float].resize(format.shape)
    }
    // for mkldnn, it always use tnc format shape even though format is ntc
    if (format.layout == Memory.Format.ntc && format.isInstanceOf[HeapData]) {
      tensor.toTensor[Float].resize(format.shape)
    }
  }

  private def createInt8PrimDesc(): Long = {
    val attr = MklDnn.CreateAttr()
    MklDnn.AttrSetIntOutputRoundMode(attr, 1)

    if (realOutput(0).scales == null || realOutput(0).scales.isEmpty) {
      realOutput(0).setMask(realInput(0).mask)
      realOutput(0).setScales(realInput(0).scales)
    }

    // if convert s8/u8 to f32, we should set the scale factor to 1.0f/x
    if (realOutput(0).dataType == DataType.F32) {
      realOutput(0).setScales(realOutput(0).scales.map(1.0f / _))
    }

    // copy the scales back to outputFormats if not equal
    if (realOutput(0) ne _outputFormats(0)) {
      _outputFormats(0).setMask(realOutput(0).mask)
      _outputFormats(0).setScales(realOutput(0).scales)
    }

    require(realOutput(0).scales.nonEmpty)
    MklDnn.AttrSetOutputScales(attr, realOutput(0).scales.length, realOutput(0).mask,
      realOutput(0).scales)

    MklDnn.ReorderPrimitiveDescCreateV2(
      realInput(0).getPrimitiveDescription(runtime),
      realOutput(0).getPrimitiveDescription(runtime),
      attr)
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = if (inputFormat == null) inputs else Array(inputFormat)
    require(_inputFormats.length == 1, "Only accept one tensor as input")

    if (outputFormat == null) _outputFormats = _inputFormats
    shapeToString(_inputFormats(0).shape)

    require(_inputFormats(0).shape.product == _outputFormats(0).shape.product,
      "input output memory not match, input shape " + shapeToString(_inputFormats(0).shape)
        + "output shape " + shapeToString(_outputFormats(0).shape))

    val inputShape = _inputFormats(0).shape
    val outputShape = _outputFormats(0).shape
    val inputLayout = _inputFormats(0).layout
    val outputLayout = _outputFormats(0).layout
    realInput = _inputFormats
    realOutput = _outputFormats

    if (inputLayout != outputLayout) {
      if (inputLayout == Memory.Format.nhwc || inputLayout == Memory.Format.ntc) {
        // remind: if format of input MemoryData is nhwc or ntc,
        // its shape should be output shape
        realInput = initMemory(_inputFormats(0), outputShape, inputLayout)
      } else if (outputLayout == Memory.Format.nhwc || outputLayout == Memory.Format.ntc) {
        // remind: if format of output MemoryData is nhwc or ntc,
        // its shape should be input shape
        realOutput = initMemory(_outputFormats(0), inputShape, outputLayout)
      }
    }

    val noInt8Formats = inputFormats()(0).dataType == DataType.F32 &&
      outputFormats()(0).dataType == DataType.F32

    val fwdReorderPrimDesc = if (noInt8Formats) {
      MklDnn.ReorderPrimitiveDescCreate(
        realInput(0).getPrimitiveDescription(runtime),
        realOutput(0).getPrimitiveDescription(runtime))
    } else {
      createInt8PrimDesc()
    }

    val fwdReorderPrim = MklDnn.PrimitiveCreate2(fwdReorderPrimDesc,
      Array(realInput(0).getPrimitive(runtime)), Array(0), 1,
      Array(realOutput(0).getPrimitive(runtime)), 1)

    updateOutputPrimitives = Array(fwdReorderPrim)

    // recover to original data
    output = initTensor(realOutput(0))

    reshapeOutputIfNeeded(_outputFormats(0), output.toTensor[Float])

    (_inputFormats, _outputFormats)
  }

  override def getUpdateGradInputMemoryPrimitives(): Array[Long] = {
    realgradOutput.map(_.getPrimitive(runtime)) ++ realgradInput.map(_.getPrimitive(runtime))
  }

  override def getUpdateOutputMemoryPrimitives(): Array[Long] = {
    realInput.map(_.getPrimitive(runtime)) ++ realOutput.map(_.getPrimitive(runtime))
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
      "gradInput and gradOutput memory not match," +
        "gradInput shape " + shapeToString(_gradInputFormats(0).shape)
        + "gradOutput shape " + shapeToString(_gradOutputFormats(0).shape))

    val gradInputShape = _gradInputFormats(0).shape
    val gradOutputShape = _gradOutputFormats(0).shape
    val gradInputLayout = _gradInputFormats(0).layout
    val gradOutputLayout = _gradOutputFormats(0).layout
    realgradInput = _gradInputFormats
    realgradOutput = _gradOutputFormats

    if (gradInputLayout != gradOutputLayout) {
      if (gradOutputLayout == Memory.Format.nhwc || gradOutputLayout == Memory.Format.ntc) {
        // remind: if format of gradOutput MemoryData is nhwc or ntc,
        // its shape should be gradInput shape
        realgradOutput = initMemory(_gradOutputFormats(0), gradInputShape, gradOutputLayout)
      } else if (gradInputLayout == Memory.Format.nhwc || gradInputLayout == Memory.Format.ntc) {
        // remind: if format of gradInput MemoryData is nhwc or ntc,
        // its shape should be gradOutput shape
        realgradInput = initMemory(_gradInputFormats(0), gradOutputShape, gradInputLayout)
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

    reshapeOutputIfNeeded(_gradInputFormats(0), gradInput.toTensor[Float])

    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = super.updateGradInput(input, gradOutput)
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
