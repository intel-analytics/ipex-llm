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

import com.intel.analytics.bigdl.mkl.{AlgKind, MklDnn, PropKind}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

class LRN(
  size: Int = 5,
  alpha: Double = 1.0,
  beta: Double = 0.75,
  k: Double = 1.0
) extends MklDnnLayer {
  private val UNDEFINED = 0

  @transient private var workSpace : Tensor[Float] = _
  @transient private var workSpaceFormat: MemoryData = _
  @transient private var fwdPrimDesc: Long = UNDEFINED
  @transient private var fwdMemPrims: Array[Long] = _
  @transient private var bwdMemPrims: Array[Long] = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val description = MklDnn.LRNForwardDescInit(
      PropKind.ForwardTraining, AlgKind.LrnAcrossChannels,
      _inputFormats(0).getMemoryDescription(), size, alpha.toFloat, beta.toFloat, k.toFloat)
    fwdPrimDesc = MklDnn.PrimitiveDescCreate(description, runtime.engine, 0L)
    _outputFormats = Array(MemoryData.primitiveOutput(fwdPrimDesc))
    workSpaceFormat = MemoryData.primitiveWorkSpace(fwdPrimDesc)
    workSpace = initTensor(workSpaceFormat)
    updateOutputPrimitives = Array(MklDnn.PrimitiveCreate2(fwdPrimDesc,
      _inputFormats.map(_.getPrimitive(runtime)), Array(0), 1, Array(_outputFormats(0),
        workSpaceFormat).map(_.getPrimitive(runtime)), 2))
    output = initTensor(_outputFormats(0))
    fwdMemPrims = Array(_inputFormats(0), _outputFormats(0), workSpaceFormat)
      .map(_.getPrimitive(runtime))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats
    val description = MklDnn.LRNBackwardDescInit(AlgKind.LrnAcrossChannels,
      _inputFormats(0).getMemoryDescription(),
      _gradOutputFormats(0).getMemoryDescription(), size, alpha.toFloat, beta.toFloat, k.toFloat)
    require(fwdPrimDesc != UNDEFINED, "You should call initFwdPrimitives first")
    val primDesc = MklDnn.PrimitiveDescCreate(description, runtime.engine, fwdPrimDesc)
    _gradInputFormats = Array(MemoryData.primitiveGradInput(primDesc))
    updateGradInputPrimitives = Array(MklDnn.PrimitiveCreate2(primDesc,
      Array(_inputFormats(0), _gradOutputFormats(0), workSpaceFormat).map(_.getPrimitive(runtime)),
      Array(0, 0, 0), 3, _gradInputFormats.map(_.getPrimitive(runtime)), 1))
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

object LRN {
  def apply(size: Int = 5, alpha: Double = 1.0, beta: Double = 0.75, k: Double = 1.0): LRN =
    new LRN(size, alpha, beta, k)
}
