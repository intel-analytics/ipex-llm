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

import com.intel.analytics.bigdl.dnnl.Memory.FormatTag
import com.intel.analytics.bigdl.dnnl.{AlgKind, ArgType, DNNL, Memory, PropKind, Query}
import com.intel.analytics.bigdl.nn.{Utils => NNUtils}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable

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
  @transient private var paddingTL: Array[Int] = _
  @transient private var paddingBR: Array[Int] = _
  @transient private var fwdPD: Long = _

  // reminder: ceilMode default value is true,
  // but in blas SpatialMaxPooling, default ceilMode is false
  private var ceilMode = true
  private val strides = Array(dW, dH)
  private val kernel = Array(kH, kW)
  private var _phase: Phase = Phase.InferencePhase


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


  /**
   * 1. Get input's MemoryData
   * 2. Get input's shape, layout and data type from its MemoryData
   * 3. Calculate output's shape, layout and data type according to input and layer type
   * 4. Initialize input & output memory descriptor
   * 5. Initialize layer operation descriptor
   * 6. Create layer operation primitive descriptor
   * 7. Create layer operation primitive
   * 8. Get output's MemoryData by querying with operation primitive descriptor
   * 9. Allocate memory for layer's output according to its MemoryData
   *
   * @param inputs
   * @param phase
   * @return
   */
  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _phase = phase
    _inputFormats = singleNativeData(inputs)

    val shape = _inputFormats(0).shape
    val (n, c, h, w) = (shape(0), shape(1), shape(2), shape(3))

    val (pt, pb, pl, pr, oh, ow) = if (padH == -1 && padW == -1) {
      val sizes = NNUtils.getSAMEOutSizeAndPadding(h, w, dH, dW, kH, kW)
      (sizes(0), sizes(1), sizes(2), sizes(3), sizes(4), sizes(5))
    } else {
      NNUtils.getPaddingAndOutputSize(h, w, dH, dW, kH, kW, padH, padW, ceilMode)
    }

    paddingTL = Array(pt, pl)
    paddingBR = Array(pb, pr)

    val propKind = if (InferencePhase == phase) {
      PropKind.ForwardScoring
    } else {
      PropKind.ForwardTraining
    }

    val inputMd = _inputFormats(0).getMemoryDescriptor()

    val outputMd = DnnlMemory.MemoryDescInit(
      4,
      Array(n, c, oh, ow),
      _inputFormats(0).dataType,
      Memory.FormatTag.any
    )

    val poolingForwardDescriptor = DnnlMemory.PoolingForwardDescInit(
      propKind,
      AlgKind.PoolingMax,
      inputMd, outputMd,
      strides, kernel,
      paddingTL, paddingBR, 0)

    fwdPD = DNNL.PrimitiveDescCreate(poolingForwardDescriptor, runtime.engine, 0L)

    _outputFormats = Array(MemoryData.primitiveOutput(fwdPD))

    val realDst = MemoryData.primitiveOutput(fwdPD)
    _outputFormats = Array(realDst)

    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> _inputFormats(0).getMemoryObject(runtime),
//      ArgType.DNNL_ARG_DST -> _outputFormats(0).getMemoryObject(runtime)
      ArgType.DNNL_ARG_DST -> realDst.getMemoryObject(runtime)
    )

    if (phase == TrainingPhase) {
      workSpaceFormat = MemoryData.operationWant(fwdPD, Query.WorkspaceMd)
      workSpaceFormat.getMemoryObject(runtime)
      workSpace = initTensor(workSpaceFormat).asInstanceOf[Tensor[Float]]
      fwdExecArgs.put(ArgType.DNNL_ARG_WORKSPACE, workSpaceFormat.getMemoryObject(runtime))
    }

    updateOutputPrimitives = Array(DnnlMemory.PrimitiveCreate(fwdPD))
    // if it's training, should have output and workspace primitive memory
    // otherwise, only need the output memory

//    output = initTensor(_outputFormats(0))
    output = initTensor(realDst)

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val shape = _inputFormats(0).shape
    val (n, c, h, w) = (shape(0), shape(1), shape(2), shape(3))

    val (pt, pb, pl, pr, oh, ow) = if (padH == -1 && padW == -1) {
      val sizes = NNUtils.getSAMEOutSizeAndPadding(h, w, dH, dW, kH, kW)
      (sizes(0), sizes(1), sizes(2), sizes(3), sizes(4), sizes(5))
    } else {
      NNUtils.getPaddingAndOutputSize(h, w, dH, dW, kH, kW, padH, padW, ceilMode)
    }

    val gradientOutputMd = DnnlMemory.MemoryDescInit(
      4,
      Array(n, c, oh, ow),
      _inputFormats(0).dataType,
      Memory.FormatTag.any
    )

    val inputMd = DnnlMemory.MemoryDescInit(
      4,
      Array(n, c, h, w),
      _inputFormats(0).dataType,
      Memory.FormatTag.any
    )

    // Pooling backward descriptor
    val backwardDescriptor = DnnlMemory.PoolingBackwardDescInit(
      AlgKind.PoolingMax,
      inputMd,
      gradientOutputMd,
      strides, kernel, paddingTL, paddingBR,
      0
    )

    // Pooling backward primitive descriptor
    val backwardPd = DnnlMemory.PrimitiveDescCreate(backwardDescriptor, runtime.engine, fwdPD)
    _gradInputFormats = Array(MemoryData.operationWant(backwardPd, Query.DiffSrcMd))
    _gradOutputFormats = Array(MemoryData.operationWant(backwardPd, Query.DiffDstMd))
    _gradOutputFormatsForWeight = _gradOutputFormats
    gradInput = initTensor(_gradInputFormats(0))
    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_DIFF_DST -> _gradOutputFormats(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_WORKSPACE -> workSpaceFormat.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_SRC -> _gradInputFormats(0).getMemoryObject(runtime)
    )

    updateGradInputPrimitives = Array(DnnlMemory.PrimitiveCreate(backwardPd))

    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
      )
      if (isTraining()) { // only for training.
        updateOutputTensors.put(ArgType.DNNL_ARG_WORKSPACE, workSpace)
      }
    }
    updateOutputTensors.put(ArgType.DNNL_ARG_SRC, input.asInstanceOf[Tensor[Float]])
    MklDnnOps.streamSubmit(updateOutputPrimitives, runtime.stream,
      fwdExecArgs, updateOutputTensors)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      updateGradInputTensors = mutable.Map(
        ArgType.DNNL_ARG_WORKSPACE -> workSpace,
        ArgType.DNNL_ARG_DIFF_SRC -> gradInput.asInstanceOf[Tensor[Float]]
      )
    }
    updateGradInputTensors.put(ArgType.DNNL_ARG_DIFF_DST, gradOutput.asInstanceOf[Tensor[Float]])
    MklDnnOps.streamSubmit(updateGradInputPrimitives,
      runtime.stream,
      bwdExecArgs,
      updateGradInputTensors)

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
