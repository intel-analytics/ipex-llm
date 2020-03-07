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

import com.intel.analytics.bigdl.dnnl.{ArgType, DNNL, DataType, Memory}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable


class ReorderMemory(inputFormat: MemoryData, outputFormat: MemoryData,
                    gradInputFormat: MemoryData, gradOutputFormat: MemoryData,
                    memoryOwner: MemoryOwner = null) extends MklDnnLayer with Releasable {
  // ReorderMemory is a special layer. It can be owned by other layers.
  // So there is an optional MemoryOwner that can be null.
  // If it is null, this means the ReorderMemory is a normal layer.
  // If it is not null, it means ReorderMemory is owned by another layer
  if (memoryOwner != null) {
    memoryOwner.registerResource(this)
  }

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
      case n: NativeData =>
        val memory = NativeData(shape, layout, src.dataType)
        memory.getMemoryObject(runtime)
        Array(memory)
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
    if (format.layout == Memory.FormatTag.nhwc && format.isInstanceOf[HeapData]) {
      tensor.toTensor[Float].resize(format.shape)
    }
    // for mkldnn, it always use tnc format shape even though format is ntc
    if (format.layout == Memory.FormatTag.ntc && format.isInstanceOf[HeapData]) {
      tensor.toTensor[Float].resize(format.shape)
    }
  }

  private def createInt8PrimDesc(inputMd: Long, outputMd: Long): Long = {
    val attr = DnnlMemory.CreateAttr()
    // TODO:
    // DNNL.AttrSetIntOutputRoundMode(attr, 1)

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
    DNNL.AttrSetOutputScales(attr, realOutput(0).scales.length,
      realOutput(0).mask, realOutput(0).scales)
    DnnlMemory.ReorderPrimitiveDescCreate(inputMd, outputMd, runtime.engine, attr)
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    // 1. Get input's MemoryData
    _inputFormats = if (inputFormat == null) inputs else Array(inputFormat)
    require(_inputFormats.length == 1, "Only accept one tensor as input")

    if (outputFormat == null) {
      _outputFormats = _inputFormats
    }

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
      if (inputLayout == Memory.FormatTag.nhwc || inputLayout == Memory.FormatTag.ntc) {
        // remind: if format of input MemoryData is nhwc or ntc,
        // its shape should be output shape
        realInput = initMemory(_inputFormats(0), outputShape, inputLayout)
      } else if (outputLayout == Memory.FormatTag.nhwc || outputLayout == Memory.FormatTag.ntc) {
        // remind: if format of output MemoryData is nhwc or ntc,
        // its shape should be input shape
        realOutput = initMemory(_outputFormats(0), inputShape, outputLayout)
      }
    }

    val noInt8Formats = inputFormats()(0).dataType == DataType.F32 &&
      outputFormats()(0).dataType == DataType.F32
    val inputMd = realInput(0).getMemoryDescriptor()
    val outputMd = realOutput(0).getMemoryDescriptor()

    val fwdReorderPrimDesc = if (noInt8Formats) {
      val engine = runtime.engine
      DnnlMemory.ReorderPrimitiveDescCreate(inputMd, outputMd, engine, 0L)
    } else {
      createInt8PrimDesc(inputMd, outputMd)
    }
    val fwdReorderPrim = DnnlMemory.PrimitiveCreate(fwdReorderPrimDesc)
    updateOutputPrimitives = Array(fwdReorderPrim)
    // recover to original data
    output = initTensor(realOutput(0))
    reshapeOutputIfNeeded(_outputFormats(0), output.toTensor[Float])

    // TODO: (realInput, realOutput)
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {

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
      if (gradOutputLayout == Memory.FormatTag.nhwc || gradOutputLayout == Memory.FormatTag.ntc) {
        // remind: if format of gradOutput MemoryData is nhwc or ntc,
        // its shape should be gradInput shape
        realgradOutput = initMemory(_gradOutputFormats(0), gradInputShape, gradOutputLayout)
      } else if (gradInputLayout == Memory.FormatTag.nhwc ||
        gradInputLayout == Memory.FormatTag.ntc) {
        // remind: if format of gradInput MemoryData is nhwc or ntc,
        // its shape should be gradOutput shape
        realgradInput = initMemory(_gradInputFormats(0), gradOutputShape, gradInputLayout)
      }
    }

    val gradOutputMd = realgradOutput(0).getMemoryDescriptor()
    val gradInputMd = realgradInput(0).getMemoryDescriptor()
    val engine = runtime.engine
    val bwdReorderPrimDesc = DnnlMemory.ReorderPrimitiveDescCreate(
      gradOutputMd, gradInputMd, engine, 0L
    )
    val bwdReorderPrim = DnnlMemory.PrimitiveCreate(bwdReorderPrimDesc)
    updateGradInputPrimitives = Array(bwdReorderPrim)
    gradInput = initTensor(realgradInput(0))
    reshapeOutputIfNeeded(_gradInputFormats(0), gradInput.toTensor[Float])

    // TODO: (realgradOutput, realgradInput)
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_FROM -> realInput(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_TO -> realOutput(0).getMemoryObject(runtime)
    )
    updateOutputTensors = mutable.Map(
      ArgType.DNNL_ARG_FROM -> input.asInstanceOf[Tensor[Float]],
      ArgType.DNNL_ARG_TO -> output.asInstanceOf[Tensor[Float]]
    )
    MklDnnOps.streamSubmit(updateOutputPrimitives,
      runtime.stream, fwdExecArgs,
      updateOutputTensors
    )
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_FROM -> realgradOutput(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_TO -> realgradInput(0).getMemoryObject(runtime)
    )
    updateGradInputTensors = mutable.Map(
      ArgType.DNNL_ARG_FROM -> gradOutput.asInstanceOf[Tensor[Float]],
      ArgType.DNNL_ARG_TO -> gradInput.asInstanceOf[Tensor[Float]]
    )
    MklDnnOps.streamSubmit(updateGradInputPrimitives,
      runtime.stream, bwdExecArgs,
      updateGradInputTensors
    )
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
  // We don't use "apply" as the function name here. The reason is that scala does not
  // allow overloaded function (functions having the same name) with default parameters
  // Hence, we bypass this issue by defining two functions.
  def create(inputFormat: MemoryData, outputFormat: MemoryData, gradInputFormat: MemoryData,
             gradOutputFomat: MemoryData)
            (implicit memoryOwner: MemoryOwner = null): ReorderMemory = {
    new ReorderMemory(inputFormat, outputFormat, gradInputFormat, gradOutputFomat, memoryOwner)
  }

  def apply(outputFormat: MemoryData, gradInputFormat: MemoryData = null)
           (implicit memoryOwner: MemoryOwner = null): ReorderMemory = {
    new ReorderMemory(null, outputFormat, gradInputFormat, null,
      memoryOwner)
  }
}