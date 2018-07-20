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

import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

class MaxPooling(
  kW: Int,
  kH: Int,
  dW: Int = 1,
  dH: Int = 1,
  padW: Int = 0,
  padH: Int = 0
) extends MklDnnLayer {
  @transient private var workSpaceFormat: MemoryData = _
  @transient private var workSpace: Tensor[Float] = _
  @transient private var fwdMemPrims: Array[Long] = _
  @transient private var bwdMemPrims: Array[Long] = _
  @transient private var paddingTL: Array[Int] = _
  @transient private var paddingBR: Array[Int] = _
  @transient private var fwdPD: Long = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)
    val n = _inputFormats(0).shape(0)
    val c = _inputFormats(0).shape(1)
    val h = _inputFormats(0).shape(2)
    val w = _inputFormats(0).shape(3)
    val (pt, pb, pl, pr, oh, ow) =
      Utils.getPaddingAndOutputSize(h, w, dH, dW, kH, kW, padH, padW)
    paddingTL = Array(pt, pl)
    paddingBR = Array(pb, pr)
          Utils.getSAMEOutSizeAndPadding(h, w, dH, dW, kH, kW)
          Utils.getOutSizeAndPaddingForDNN(h, w, dH, dW, kH, kW, padH, padW, true)
    val outputMD = MklDnn.MemoryDescInit(4, Array(n, c, oh, ow), DataType.F32, Memory.Format.any)
    val description = MklDnn.PoolingForwardDescInit(
      PropKind.Forward, AlgKind.PoolingMax,
      _inputFormats(0).getMemoryDescription(), outputMD, strides, kernel, paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)
    fwdPD = MklDnn.PrimitiveDescCreate(description, runtime.engine, 0L)
    _outputFormats = Array(MemoryData.primitiveOutput(fwdPD))
    output = initTensor(_outputFormats(0))
    workSpaceFormat = MemoryData.primitiveWorkSpace(fwdPD)
    workSpace = initTensor(workSpaceFormat)
    updateOutputPrimitives = Array(MklDnn.PrimitiveCreate2(fwdPD,
      _inputFormats.map(_.getPrimitive(runtime)), Array(0), 1,
      Array(_outputFormats(0), workSpaceFormat).map(_.getPrimitive(runtime)), 2))
    fwdMemPrims = Array(_inputFormats(0), _outputFormats(0), workSpaceFormat)
      .map(_.getPrimitive(runtime))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats
    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)
    val description = MklDnn.PoolingBackwardDescInit(AlgKind.PoolingMax,
      _inputFormats(0).getMemoryDescription(),
      _gradOutputFormats(0).getMemoryDescription(),
      strides, kernel, paddingTL, paddingBR, MklDnn.PaddingKind.mkldnnPaddingZero)

    val pd = MklDnn.PrimitiveDescCreate(description, runtime.engine, fwdPD)
    _gradInputFormats = Array(MemoryData.primitiveGradInput(pd))
    updateGradInputPrimitives = Array(MklDnn.PrimitiveCreate2(pd,
      Array(_gradOutputFormats(0), workSpaceFormat).map(_.getPrimitive(runtime)),
      Array(0, 0), 2, _gradInputFormats.map(_.getPrimitive(runtime)), 1))
    gradInput = initTensor(_gradInputFormats(0))
    bwdMemPrims = Array(_inputFormats(0), _gradOutputFormats(0), workSpaceFormat,
      _gradInputFormats(0)).map(_.getPrimitive(runtime))
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    val buffer = Array(input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
      workSpace)
    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, 1, fwdMemPrims, buffer)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    val buffer = Array(
      input.asInstanceOf[Tensor[Float]], gradOutput.asInstanceOf[Tensor[Float]], workSpace,
      gradInput.asInstanceOf[Tensor[Float]])
    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives, 1,
      bwdMemPrims, buffer)
    gradInput
  }
}

object MaxPooling {
  def apply(
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0
  ): MaxPooling = new MaxPooling(kW, kH, dW, dH, padW, padH)
}
