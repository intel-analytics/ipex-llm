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
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer

class ConcatTable extends MklDnnContainer with MklInt8Convertible {

  output = T()

  @transient private var sumPrimitive: Array[Long] = null
  @transient private var tensors: Array[Tensor[Float]] = null
  @transient private var tensorPrimitives: Array[Long] = null

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
      tensors(i) = modules(i).updateGradInput(input, gradOutput.toTable(i + 1))
        .asInstanceOf[Tensor[Float]]
      i += 1
    }
    MklDnnOps.streamSubmit(runtime.stream, 1, sumPrimitive, 1, tensorPrimitives, tensors)
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

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(mklDnnModules != null, "You should call compile first")
    require(inputs.length == 1, "Concat only accept one tensor")
    val buffer = new ArrayBuffer[MemoryData]()
    mklDnnModules.foreach(m => {
      val (realInput, out) = m.initFwdPrimitives(inputs, phase)
      require(out.length == 1, "output should be one tensor")
      inputs.zip(realInput).map {case(f, t) => reorderManager.register(f, t)}
      buffer.append(out(0))
    })
    _outputFormats = buffer.toArray
    _inputFormats = inputs
    (inputs, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == mklDnnModules.length, "grad tensor number is not correct")
    _gradOutputFormats = new Array[MemoryData](grads.length)
    val subGradInputs = new Array[MemoryData](grads.length)
    tensorPrimitives = new Array[Long](grads.length + 1)
    var shape: Array[Int] = null
    for(i <- 0 until grads.length) {
      val m = mklDnnModules(i)
      val (realGrads, gradInput) = m.initBwdPrimitives(Array(grads(i)), phase)
      require(realGrads.length == 1, "real grad length should be 1")
      _gradOutputFormats(i) = realGrads(0)
      require(gradInput.length == 1, "real grad length should be 1")
      subGradInputs(i) = gradInput(0)
      tensorPrimitives(i) = gradInput(0).getPrimitive(runtime)
      if (shape == null) {
        shape = gradInput(0).shape.clone()
      } else {
        require(shape.length == gradInput(0).shape.length, "backward grad shape should be same")
        for(j <- 0 until shape.length) {
          require(shape(j) == gradInput(0).shape(j), "backward grad shape size should be same")
        }
      }
    }
    val outputMD = MklDnnMemory.MemoryDescInit(shape.length, shape, DataType.F32, Memory.Format.any)
    val scales = grads.map(_ => 1f)
    val pd = MklDnnMemory.SumPrimitiveDescCreate(outputMD, grads.length, scales,
      subGradInputs.map(_.getPrimitiveDescription(runtime)))
    _gradInputFormats = Array(MemoryData.primitiveOutput(pd))
    tensorPrimitives(grads.length) = _gradInputFormats(0).getPrimitive(runtime)
    sumPrimitive = Array(MklDnnMemory.PrimitiveCreate2(pd,
      subGradInputs.map(_.getPrimitive(runtime)), new Array[Int](grads.length),
      grads.length, _gradInputFormats.map(_.getPrimitive(runtime)), 1))
    gradInput = initTensor(_gradInputFormats(0))
    tensors = new Array[Tensor[Float]](grads.length + 1)
    tensors(grads.length) = gradInput.asInstanceOf[Tensor[Float]]
    (_gradOutputFormats, _gradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grads: Array[MemoryData], phase: Phase) = {
    val realGradsBuffer = new ArrayBuffer[MemoryData]()
    for(i <- 0 until grads.length) {
      val m = mklDnnModules(i)
      val realGradOutput = m.initGradWPrimitives(Array(grads(i)), phase)
      require(realGradOutput.length == 1, s"real grad length should be 1, " +
        s"but it's ${realGradOutput.length}")
      realGradsBuffer.append(realGradOutput(0))
    }
    _gradOutputWeightFormats = realGradsBuffer.toArray
    _gradOutputWeightFormats
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

  private var _inputFormats: Array[MemoryData] = _
  private var _gradInputFormats: Array[MemoryData] = _
  private var _outputFormats: Array[MemoryData] = _
  private var _gradOutputFormats: Array[MemoryData] = _
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
