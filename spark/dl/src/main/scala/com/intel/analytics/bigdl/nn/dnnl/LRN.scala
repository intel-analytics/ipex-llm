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
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable

class LRN(
  size: Int = 5,
  alpha: Double = 1.0,
  beta: Double = 0.75,
  k: Double = 1.0
) extends MklDnnLayer {
  private val UNDEFINED = 0

  @transient private var workSpaceFormat: MemoryData = _
  @transient private var fwdPrimDesc: Long = UNDEFINED
  @transient private var fwdMemPrims: Array[Long] = _
  @transient private var bwdMemPrims: Array[Long] = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    // the lrn only support f32
    _inputFormats = Array(NativeData(inputs(0).shape, inputs(0).layout, DataType.F32))

    val kind = if (phase == InferencePhase) {
      PropKind.ForwardScoring
    } else {
      PropKind.ForwardTraining
    }

    val description = DnnlMemory.LRNForwardDescInit(
      kind, AlgKind.LrnAcrossChannels,
      _inputFormats(0).getMemoryDescriptor(),
      size, alpha.toFloat, beta.toFloat, k.toFloat)

    fwdPrimDesc = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, 0L)
    _outputFormats = Array(MemoryData.primitiveOutput(fwdPrimDesc))
    output = initTensor(_outputFormats(0))

    fwdMemPrims = if (phase == InferencePhase) {
      Array(_inputFormats(0), _outputFormats(0)).map(_.getMemoryObject(runtime))
    } else {
      // we only create the workspace when the phase is training
      workSpaceFormat = MemoryData.operationWant(fwdPrimDesc, Query.WorkspaceMd, 0)
      Array(_inputFormats(0), _outputFormats(0)).map(_.getMemoryObject(runtime))
    }

    fwdExecArgs = mutable.Map (
      ArgType.DNNL_ARG_SRC -> inputFormats().head.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DST -> _outputFormats.head.getMemoryObject(runtime)
    )
    updateOutputPrimitives = Array(DnnlMemory.PrimitiveCreate(fwdPrimDesc))

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats
    val description = DnnlMemory.LRNBackwardDescInit(AlgKind.LrnAcrossChannels,
      _inputFormats(0).getMemoryDescriptor(),
      _gradOutputFormats(0).getMemoryDescriptor(), size, alpha.toFloat, beta.toFloat, k.toFloat)
    require(fwdPrimDesc != UNDEFINED, "You should call initFwdPrimitives first")
    val primDesc = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, fwdPrimDesc)
    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcMd))
    updateGradInputPrimitives = Array(DnnlMemory.PrimitiveCreate(primDesc))
    gradInput = initTensor(_gradInputFormats(0))
    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> inputFormats().head.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_DST -> gradOutputFormats().head.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_SRC -> gradInputFormats().head.getMemoryObject(runtime)
    )
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
      )
    }
    updateWithNewTensor(updateOutputTensors, ArgType.DNNL_ARG_SRC, input)
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
    MklDnnOps.streamSubmit(updateGradInputPrimitives, runtime.stream,
      bwdExecArgs, updateGradInputTensors)
    gradInput
  }
}

object LRN {
  def apply(size: Int = 5, alpha: Double = 1.0, beta: Double = 0.75, k: Double = 1.0): LRN =
    new LRN(size, alpha, beta, k)
}
