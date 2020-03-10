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
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable

class ReLU(value: Float = 0.0f) extends MklDnnLayer with MklInt8Convertible {
  private val UNDEFINED: Long = 0

  @transient private var fwdPrimDesc: Long = UNDEFINED

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val description = DnnlMemory.EltwiseForwardDescInit(
      PropKind.Forward, AlgKind.EltwiseRelu, _inputFormats(0).getMemoryDescriptor(), value, 0)
    fwdPrimDesc = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, 0L)
    _outputFormats = Array(MemoryData.primitiveOutput(fwdPrimDesc))
    updateOutputPrimitives = Array(
      DnnlMemory.PrimitiveCreate(fwdPrimDesc)
    )
    initFwdExecArgs()
    output = initTensor(_outputFormats(0))

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats

    val description = DnnlMemory.EltwiseBackwardDescInit(AlgKind.EltwiseRelu,
      _gradOutputFormats(0).getMemoryDescriptor(), _inputFormats(0).getMemoryDescriptor(),
      value, 0)
    require(fwdPrimDesc != UNDEFINED, "You should call initFwdPrimitives first")
    val primDesc = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, fwdPrimDesc)
    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffDstMd))
    updateGradInputPrimitives = Array(
      DnnlMemory.PrimitiveCreate(primDesc))
    gradInput = initTensor(_gradInputFormats(0))
    (_gradOutputFormats, _gradInputFormats)
  }

   override def initBwdExecArgs(): Unit = {
    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_DIFF_SRC ->
        gradInputFormats().map(_.getMemoryObject(runtime)).head,
      ArgType.DNNL_ARG_DIFF_DST ->
        gradOutputFormats().map(_.getMemoryObject(runtime)).head,
      ArgType.DNNL_ARG_SRC ->
        inputFormats().map(_.getMemoryObject(runtime)).head
    )
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (bwdExecArgs == null) {
      initBwdExecArgs()
    }

    if (updateGradInputTensors == null) {
      updateGradInputTensors = mutable.Map(
        ArgType.DNNL_ARG_DIFF_SRC ->
          gradInput.asInstanceOf[Tensor[Float]]
      )
    }

    updateGradInputTensors(ArgType.DNNL_ARG_SRC) = input.asInstanceOf[Tensor[Float]]
    updateGradInputTensors(ArgType.DNNL_ARG_DIFF_DST) = gradOutput.asInstanceOf[Tensor[Float]]

    MklDnnOps.streamSubmit(updateGradInputPrimitives,
      runtime.stream, bwdExecArgs,
      updateGradInputTensors
    )

    gradInput
  }
}

object ReLU {
  def apply(value: Float = 0.0f): ReLU = new ReLU(value)
}
