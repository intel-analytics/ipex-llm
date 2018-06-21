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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer

class ConcatTable extends MklDnnContainer {

  output = T()

  override def updateOutput(input: Activity): Activity = {
    require(modules.length > 0, "empty modules of concat table")
    var i = 0
    while (i < modules.length) {
      val currentOutput = modules(i).forward(
        reorderManager.infer(_inputFormats, mklDnnModules(i).inputFormats(), input))
      output.toTable(i + 1) = currentOutput
      i += 1
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    require(modules.length > 0, "empty modules of concat table")

    var i = 0
    while (i < modules.length) {
      val currentGradInput = modules(i).updateGradInput(input, gradOutput.toTable(i + 1))
        .asInstanceOf[Tensor[Float]]
      if (i == 0) {
        gradInput.toTensor[Float].resizeAs(currentGradInput).copy(currentGradInput)
      } else {
        gradInput.toTensor[Float].add(currentGradInput)
      }
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < modules.length) {
      modules(i).accGradParameters(input, gradOutput.toTable(i + 1))
      i += 1
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(mklDnnModules != null, "You should call compile first")
    require(inputs.length == 1, "Concat only accept one tensor")
    val buffer = new ArrayBuffer[MemoryData]()
    mklDnnModules.foreach(m => {
      val (realInput, out) = m.initFwdPrimitives(inputs, phase)
      require(out.length == 1, "output should be one tensor")
      reorderManager.register(inputs(0), realInput(0))
      buffer.append(out(0))
    })
    _outputFormats = buffer.toArray
    _inputFormats = inputs
    (inputs, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == mklDnnModules.length, "grad tensor number is not correct")
    val realGradsBuffer = new ArrayBuffer[MemoryData]()
    for(i <- 0 until grads.length) {
      val m = mklDnnModules(i)
      val (realGrads, gradInput) = m.initBwdPrimitives(Array(grads(i)), phase)
      require(realGrads.length == 1, "real grad length should be 1")
      realGradsBuffer.append(realGrads(0))
      require(gradInput.length == 1, "real grad length should be 1")
      if (_gradInputFormats == null) {
        _gradInputFormats = gradInput
      } else {
        require(_gradInputFormats(0) == gradInput(0), "reorder backward should be same")
      }
    }
    _gradOutputFormats = realGradsBuffer.toArray
    (realGradsBuffer.toArray, _gradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grads: Array[MemoryData], phase: Phase) = {
    val realGradsBuffer = new ArrayBuffer[MemoryData]()
    for(i <- 0 until grads.length) {
      val m = mklDnnModules(i)
      val realGradOutput = m.initGradWPrimitives(grads, phase)
      require(realGradOutput.length == 1, "real grad length should be 1")
      realGradsBuffer.append(realGradOutput(0))
    }
    _gradOutputWeightFormats = realGradsBuffer.toArray
    _gradOutputWeightFormats
  }

  override private[mkldnn] def inputFormats() = {
    require(_inputFormats != null, "You should call initFwdPrimitives first")
    _inputFormats
  }

  override private[mkldnn] def gradInputFormats() = {
    require(_gradInputFormats != null, "You should call initBwdPrimitives first")
    _gradInputFormats
  }

  override private[mkldnn] def outputFormats() = {
    require(_outputFormats != null, "You should call initFwdPrimitives first")
    _outputFormats
  }

  override private[mkldnn] def gradOutputFormats() = {
    require(_gradInputFormats != null, "You should call initBwdPrimitives first")
    _gradOutputFormats
  }

  private var _inputFormats: Array[MemoryData] = _
  private var _gradInputFormats: Array[MemoryData] = _
  private var _outputFormats: Array[MemoryData] = _
  private var _gradOutputFormats: Array[MemoryData] = _
  private var _gradOutputWeightFormats: Array[MemoryData] = _

  override private[mkldnn] def gradOutputWeightFormats() = _gradOutputWeightFormats
}

object ConcatTable {
  def apply(): ConcatTable = new ConcatTable()
}
