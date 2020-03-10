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

import com.intel.analytics.bigdl.dnnl.{AlgKind, ArgType, DNNL, DataType, Memory}
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable


class CAddTable extends MklDnnLayer with MklInt8Convertible {

  // TODO: should CAddTable be defined as a MklDnnContainer?
  protected var subModuleFwdPrimitives: Array[Array[Long]] = _

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    subModuleFwdPrimitives = new Array[Array[Long]](inputs.length)
    val headInputsFormats = inputs.head
    _inputFormats = nativeData(inputs)
    _outputFormats = Array(MemoryData.cloneFormatWithDesc(inputFormats().head))

    for(i <- 0 until inputs.length) {
      require(headInputsFormats.shape.length == inputs(i).shape.length, "dimension not match")
      for(j <- 0 until headInputsFormats.shape.length) {
        require(headInputsFormats.shape(j) == inputs(i).shape(j), "size not match")
      }
      val currInputFormat = inputs(i)
      val binaryDescInit = DNNL.BinaryDescInit(AlgKind.BinaryAdd,
        outputFormats().head.getMemoryDescriptor(),
        currInputFormat.getMemoryDescriptor(),
        outputFormats().head.getMemoryDescriptor()
      )
      val binaryPd = DnnlMemory.PrimitiveDescCreate(binaryDescInit, runtime.engine, 0)
      subModuleFwdPrimitives(i) = Array(DNNL.PrimitiveCreate(binaryPd))
    }

    _outputFormats.head.getMemoryObject(runtime)
    output = initTensor(_outputFormats.head)

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = grad
    _gradOutputFormatsForWeight = grad
    _gradInputFormats = new Array[MemoryData](_inputFormats.length).map(a => grad(0))
    gradInput = T()
    (_gradOutputFormats, _gradInputFormats)
  }


  override def updateOutput(input: Activity): Activity = {
    output.toTensor[Float].zero()

    for (i <- 0 until input.toTable.length()) {
      fwdExecArgs = mutable.Map(
        ArgType.DNNL_ARG_SRC_0 -> outputFormats().head.getMemoryObject(runtime),
        ArgType.DNNL_ARG_SRC_1 -> inputFormats()(i).getMemoryObject(runtime),
        ArgType.DNNL_ARG_DST -> outputFormats().head.getMemoryObject(runtime)
      )
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC_0 -> output.toTensor[Float],
        ArgType.DNNL_ARG_SRC_1 -> input.toTable(i + 1).asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DST -> output.toTensor[Float]
      )
      MklDnnOps.streamSubmit(subModuleFwdPrimitives(i), runtime.stream,
        fwdExecArgs, updateOutputTensors)
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    require(gradOutput.isTensor, "gradOutput should be a tensor")
    val _gradInput = gradInput.toTable
    var i = 1
    while(i <= _inputFormats.length) {
      _gradInput(i) = gradOutput
      i += 1
    }
    gradInput
  }
}

object CAddTable {
  def apply(): CAddTable = new CAddTable()
}