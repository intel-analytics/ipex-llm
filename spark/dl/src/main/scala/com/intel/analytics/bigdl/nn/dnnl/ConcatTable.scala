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
import com.intel.analytics.bigdl.dnnl.{AlgKind, ArgType, DNNL, Memory}
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


class ConcatTable extends MklDnnContainer with MklInt8Convertible {

  output = T()

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(mklDnnModules != null, "You should call compile first")
    require(inputs.length == 1, "Concat only accept one tensor")
    val buffer = new ArrayBuffer[MemoryData]()
    var prevSubModule: MklDnnModule = null

    for(i <- 0 until mklDnnModules.length) {
      val currSubModule: MklDnnModule = mklDnnModules(i)
      currSubModule.initFwdPrimitives(inputs, phase)
      val subModuleInFormat = currSubModule.inputFormats()
      val subModuleOutFormat = currSubModule.outputFormats()
      require(subModuleOutFormat.length == 1, "output should be one tensor")
      inputs.zip(subModuleInFormat).map {case(f, t) => reorderManager.register(f, t)}
      buffer.append(subModuleOutFormat(0))
      if(prevSubModule != null) {
        require(prevSubModule.inputFormats().head.shape sameElements
          currSubModule.inputFormats().head.shape)
      }
      prevSubModule = currSubModule
    }
    _outputFormats = buffer.toArray
    _inputFormats = inputs

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == mklDnnModules.length, "grad tensor number is not correct")
    _gradOutputFormats = new Array[MemoryData](grads.length)
    _gradInputFormats = new Array[MemoryData](inputFormats().length)
    subModuleBwdPrimitives = new Array[Array[Long]](grads.length)

    for(i <- 0 until grads.length) {
      val subModule = mklDnnModules(i)
      subModule.initBwdPrimitives(Array(grads(i)), phase)
      val subGradOutFormats = subModule.gradOutputFormats()
      val subGradInFormats = subModule.gradInputFormats()
      require(subGradOutFormats.length == 1, "real grad length should be 1")
      require(subGradInFormats.length == 1, "real grad length should be 1")
      _gradOutputFormats(i) = subGradOutFormats.head
      if (i == 0) {
        _gradInputFormats(0) = NativeData(
          subGradInFormats.head.shape,
          subGradInFormats.head.layout,
          subGradInFormats.head.dataType
        )
        _gradInputFormats = Array(MemoryData.cloneFormatWithDesc(
          mklDnnModules.head.gradInputFormats().head))
        _gradInputFormats.map(_.getMemoryObject(runtime))
        gradInput = initTensor(_gradInputFormats.head)
      }
      val binaryDescInit = DNNL.BinaryDescInit(
        AlgKind.BinaryAdd,
        gradInputFormats().head.getMemoryDescriptor(),
        subModule.gradInputFormats().head.getMemoryDescriptor(),
        gradInputFormats().head.getMemoryDescriptor()
      )
      val fwdPd = DnnlMemory.PrimitiveDescCreate(binaryDescInit, runtime.engine, 0)
      subModuleBwdPrimitives(i) = Array(DNNL.PrimitiveCreate(fwdPd))
    }

    (_gradOutputFormats, _gradInputFormats)
  }


  override private[mkldnn] def initGradWPrimitives(grads: Array[MemoryData], phase: Phase) = {
    val realGradsBuffer = new ArrayBuffer[MemoryData]()
    for(i <- 0 until grads.length) {
      val m = mklDnnModules(i)
      val realGradOutput = m.initGradWPrimitives(Array(grads(i)), phase)
      realGradsBuffer.append(realGradOutput(0))
    }
    _gradOutputWeightFormats = realGradsBuffer.toArray
    _gradOutputWeightFormats
  }

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
    gradInput.toTensor[Float].zero()
    require(modules.length > 0, "empty modules of concat table")
    modules.zipWithIndex.map {
      case (m, i) =>
        val subGradInTensor = m.updateGradInput(input, gradOutput.toTable(i + 1))
        bwdExecArgs = mutable.Map(
          ArgType.DNNL_ARG_SRC_0 -> gradInputFormats().head.getMemoryObject(runtime),
          ArgType.DNNL_ARG_SRC_1 -> mklDnnModules(i).gradInputFormats().head
            .getMemoryObject(runtime),
          ArgType.DNNL_ARG_DST -> gradInputFormats().head.getMemoryObject(runtime)
        )
        updateGradInputTensors = mutable.Map(
          ArgType.DNNL_ARG_SRC_0 -> gradInput.toTensor[Float],
          ArgType.DNNL_ARG_SRC_1 -> subGradInTensor.asInstanceOf[Tensor[Float]],
          ArgType.DNNL_ARG_DST -> gradInput.toTensor[Float]
        )
        MklDnnOps.streamSubmit(subModuleBwdPrimitives(i), runtime.stream,
          bwdExecArgs, updateGradInputTensors)
    }
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < modules.length) {
      modules(i).accGradParameters(input, gradOutput.toTable(i + 1))
      modules(i).asyncGradient
      i += 1
    }
  }

  private[mkldnn] def reconstruct(): Unit = {
    mklDnnModules = modules.map(_.asInstanceOf[MklDnnModule]).toArray
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
    require(_gradOutputFormats != null, "You should call initBwdPrimitives first")
    _gradOutputFormats
  }

  private var _gradOutputWeightFormats: Array[MemoryData] = _

  override private[mkldnn] def gradOutputWeightFormats() = _gradOutputWeightFormats

  override def toString(): String = {
    val tab = "\t"
    val line = "\n"
    val next = "  |`-> "
    val lastNext = "   `-> "
    val ext = "  |    "
    val extlast = "       "
    val last = "   ... -> "
    var str = s"${getPrintName}"
    str = str + " {" + line + tab + "input"
    var i = 1
    while (i <= modules.length) {
      if (i == modules.length) {
        str = str + line + tab + lastNext + "(" + i + "): " +
          modules(i-1).toString.replace(line, line + tab + extlast)
      } else {
        str = str + line + tab + next + "(" + i + "): " +
          modules(i-1).toString.replace(line, line + tab + ext)
      }
      i += 1
    }
    str = str + line + tab + last + "output"
    str = str + line + "}"
    str
  }
}

object ConcatTable {
  def apply(): ConcatTable = new ConcatTable()
}