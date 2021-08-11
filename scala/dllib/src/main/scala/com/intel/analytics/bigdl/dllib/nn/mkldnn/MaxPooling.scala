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
import com.intel.analytics.bigdl.nn.{Utils => NNUtils}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor

class MaxPooling(
  kW: Int,
  kH: Int,
  dW: Int = 1,
  dH: Int = 1,
  padW: Int = 0,
  padH: Int = 0,
  val format: DataFormat = DataFormat.NCHW
) extends MklDnnLayer {
  @transient private var workSpaceFormat: MemoryData = _
  @transient private var workSpace: Tensor[Float] = _
  @transient private var fwdMemPrims: Array[Long] = _
  @transient private var bwdMemPrims: Array[Long] = _
  @transient private var paddingTL: Array[Int] = _
  @transient private var paddingBR: Array[Int] = _
  @transient private var fwdPD: Long = _

  // reminder: ceilMode default value is true,
  // but in blas SpatialMaxPooling, default ceilMode is false
  private var ceilMode = true

  /**
   * set ceil mode
   * @return this
   */
  def ceil(): MaxPooling = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): MaxPooling = {
    ceilMode = false
    this
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)
    val n = _inputFormats(0).shape(0)
    val c = _inputFormats(0).shape(1)
    val h = _inputFormats(0).shape(2)
    val w = _inputFormats(0).shape(3)

    val (pt, pb, pl, pr, oh, ow) = if (padH == -1 && padW == -1) {
      val sizes = NNUtils.getSAMEOutSizeAndPadding(h, w, dH, dW, kH, kW)
      (sizes(0), sizes(1), sizes(2), sizes(3), sizes(4), sizes(5))
    } else {
      NNUtils.getPaddingAndOutputSize(h, w, dH, dW, kH, kW, padH, padW, ceilMode)
    }
    paddingTL = Array(pt, pl)
    paddingBR = Array(pb, pr)

    val kind = if (InferencePhase == phase) {
      PropKind.ForwardScoring
    } else {
      PropKind.ForwardTraining
    }

    val outputMD = MklDnnMemory.MemoryDescInit(4, Array(n, c, oh, ow), inputs(0).dataType,
      Memory.Format.any)
    val description = MklDnnMemory.PoolingForwardDescInit(
      kind, AlgKind.PoolingMax,
      _inputFormats(0).getMemoryDescription(), outputMD, strides, kernel, paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)
    fwdPD = MklDnnMemory.PrimitiveDescCreate(description, runtime.engine, 0L)

    _outputFormats = Array(MemoryData.primitiveOutput(fwdPD))
    output = initTensor(_outputFormats(0))
    if (phase == TrainingPhase) {
      workSpaceFormat = MemoryData.operationWant(fwdPD, Query.WorkspacePd)
      workSpace = initTensor(workSpaceFormat).asInstanceOf[Tensor[Float]]
      fwdMemPrims = Array(_inputFormats(0), _outputFormats(0), workSpaceFormat)
        .map(_.getPrimitive(runtime))
    } else {
      fwdMemPrims = Array(_inputFormats(0), _outputFormats(0)).map(_.getPrimitive(runtime))
    }

    updateOutputPrimitives = Array(MklDnnMemory.PrimitiveCreate2(fwdPD,
      _inputFormats.map(_.getPrimitive(runtime)), Array(0), 1,
      fwdMemPrims.drop(1), fwdMemPrims.length - 1))
    // if it's training, should have output and workspace primitive memory
    // otherwise, only need the output memory

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats
    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)
    val description = MklDnnMemory.PoolingBackwardDescInit(AlgKind.PoolingMax,
      _inputFormats(0).getMemoryDescription(),
      _gradOutputFormats(0).getMemoryDescription(),
      strides, kernel, paddingTL, paddingBR, MklDnn.PaddingKind.mkldnnPaddingZero)

    val pd = MklDnnMemory.PrimitiveDescCreate(description, runtime.engine, fwdPD)
    _gradInputFormats = Array(MemoryData.operationWant(pd, Query.DiffSrcPd))
    updateGradInputPrimitives = Array(MklDnnMemory.PrimitiveCreate2(pd,
      Array(_gradOutputFormats(0), workSpaceFormat).map(_.getPrimitive(runtime)),
      Array(0, 0), 2, _gradInputFormats.map(_.getPrimitive(runtime)), 1))
    gradInput = initTensor(_gradInputFormats(0))
    bwdMemPrims = Array(_inputFormats(0), _gradOutputFormats(0), workSpaceFormat,
      _gradInputFormats(0)).map(_.getPrimitive(runtime))
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    val buffer = if (fwdMemPrims.length == 3) { // only for training.
      Array(input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]],
        workSpace)
    } else {
      Array(input.asInstanceOf[Tensor[Float]], output.asInstanceOf[Tensor[Float]])
    }

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, 1, fwdMemPrims,
      buffer)
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
    padH: Int = 0,
    format: DataFormat = DataFormat.NCHW
  ): MaxPooling = new MaxPooling(kW, kH, dW, dH, padW, padH, format)
}
