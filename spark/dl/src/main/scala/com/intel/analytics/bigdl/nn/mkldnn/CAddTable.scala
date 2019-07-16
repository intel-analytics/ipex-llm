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
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils.T

class CAddTable extends MklDnnLayer with MklInt8Convertible {
  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = nativeData(inputs)
    val shape = inputs(0).shape.clone()
    for(i <- 1 until inputs.length) {
      require(shape.length == inputs(i).shape.length, "dimension not match")
      for(j <- 0 until shape.length) {
        require(shape(j) == inputs(i).shape(j), "size not match")
      }
    }

    val outputMD = MklDnnMemory.MemoryDescInit(shape.length, shape,
      inputs(0).dataType, Memory.Format.any)

    val scales = inputs.map { x =>
      if (x.dataType != DataType.F32 && x.scales.nonEmpty) {
        // here only supports 1 scale for cadd
        val max = inputs.flatMap(_.scales).max
        x.scales.head / max
      } else {
        1.0f
      }
    }

    val pd = MklDnnMemory.SumPrimitiveDescCreate(outputMD, inputs.length, scales,
      inputs.map(_.getPrimitiveDescription(runtime)))
    _outputFormats = Array(MemoryData.primitiveOutput(pd))
    updateOutputPrimitives = Array(MklDnnMemory.PrimitiveCreate2(pd,
      _inputFormats.map(_.getPrimitive(runtime)), new Array[Int](inputs.length),
      _inputFormats.length, _outputFormats.map(_.getPrimitive(runtime)), 1))
    output = initTensor(_outputFormats(0))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = grad
    _gradOutputFormatsForWeight = grad
    _gradInputFormats = new Array[MemoryData](_inputFormats.length).map(a => grad(0))
    gradInput = T()
    (_gradOutputFormats, _gradInputFormats)
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
