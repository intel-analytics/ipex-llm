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

import com.intel.analytics.bigdl.dnnl._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Utils => NNUtils}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable


class AvgPooling(
  var kW: Int,
  var kH: Int,
  dW: Int = 1,
  dH: Int = 1,
  padW: Int = 0,
  padH: Int = 0,
  globalPooling: Boolean = false
) extends MklDnnLayer {

  @transient private var workSpaceFormat: MemoryData = _
  @transient private var workSpace: Tensor[Float] = _
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
  def ceil(): AvgPooling = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): AvgPooling = {
    ceilMode = false
    this
  }

  private val algKind = if (padH == -1 && padW == -1) {
    AlgKind.PoolingAvgIncludePadding
  } else {
    AlgKind.PoolingAvgExcludePadding
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val inShape = _inputFormats(0).shape
    val (n, c, h, w) = (inShape(0), inShape(1), inShape(2), inShape(3))

    // global average pooling reduce each feature map to a single average value
    if (globalPooling) {
      kH = h
      kW = w
    }

    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)

    val (pt, pb, pl, pr, oh, ow) = if (padH == -1 && padW == -1) {
      val sizes = NNUtils.getSAMEOutSizeAndPadding(h, w, dH, dW, kH, kW)
      (sizes(0), sizes(1), sizes(2), sizes(3), sizes(4), sizes(5))
    } else {
      NNUtils.getPaddingAndOutputSize(h, w, dH, dW, kH, kW, padH, padW, ceilMode)
    }

    paddingTL = Array(pt, pl)
    paddingBR = Array(pb, pr)

    val inputMd = inputFormats()(0).getMemoryDescriptor()
    val outputMD = DnnlMemory.MemoryDescInit(4, Array(n, c, oh, ow), inputs(0).dataType,
      Memory.FormatTag.any)

    val kind = if (phase == InferencePhase) {
      PropKind.ForwardScoring
    } else {
      PropKind.ForwardTraining
    }

    val description = DnnlMemory.PoolingForwardDescInit(
      kind, algKind,
      inputMd, outputMD,
      strides, kernel, paddingTL, paddingBR,
      DNNL.PaddingKind.mkldnnPaddingZero)

    fwdPD = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, 0L)

    _outputFormats = Array(MemoryData.primitiveOutput(fwdPD))

    fwdExecArgs = mutable.Map[Int, Long](
      ArgType.DNNL_ARG_SRC -> _inputFormats(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_DST -> _outputFormats(0).getMemoryObject(runtime)
    )

    if (phase == TrainingPhase) {
      workSpaceFormat = MemoryData.operationWant(fwdPD, Query.WorkspaceMd)
      workSpaceFormat.getMemoryObject(runtime)
    }

    output = initTensor(_outputFormats(0))

    updateOutputPrimitives = Array(DnnlMemory.PrimitiveCreate(fwdPD))

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats

    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)

    val description = DnnlMemory.PoolingBackwardDescInit(algKind,
      _inputFormats(0).getMemoryDescriptor(),
      _gradOutputFormats(0).getMemoryDescriptor(),
      strides, kernel, paddingTL, paddingBR, DNNL.PaddingKind.mkldnnPaddingZero)

    val pd = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, fwdPD)

    _gradInputFormats = Array(MemoryData.operationWant(pd, Query.DiffSrcMd))

    updateGradInputPrimitives = Array(DnnlMemory.PrimitiveCreate(pd))

    gradInput = initTensor(_gradInputFormats(0))

    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> _inputFormats(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_DST -> _gradOutputFormats(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_SRC -> _gradInputFormats(0).getMemoryObject(runtime)
    )

    updateGradInputPrimitives = Array(DnnlMemory.PrimitiveCreate(pd))

    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    updateOutputTensors = mutable.Map(
      ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
      ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
    )

    MklDnnOps.streamSubmit(updateOutputPrimitives, runtime.stream,
      fwdExecArgs, updateOutputTensors)

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    updateGradInputTensors = mutable.Map(
      ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
      ArgType.DNNL_ARG_DIFF_DST -> gradOutput.asInstanceOf[Tensor[Float]],
      ArgType.DNNL_ARG_DIFF_SRC -> gradInput.asInstanceOf[Tensor[Float]]
    )

    MklDnnOps.streamSubmit(updateGradInputPrimitives,
      runtime.stream,
      bwdExecArgs,
      updateGradInputTensors)

    gradInput
  }

}

object AvgPooling {
  def apply(
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    globalPooling: Boolean = false
  ): AvgPooling = new AvgPooling(kW, kH, dW, dH, padW, padH, globalPooling)
}
