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
      val currentOutput = modules(i).forward(input)
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

  /**
   * Compute the output formats based on the input formats
   */
  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]) = {
    require(shapes.length == 1, "Concat only accept one tensor")
    require(mklDnnModules.length > 0, "Concat should contains at least one module")

    val outputShape = new ArrayBuffer[Array[Int]]()
    for(i <- 0 until mklDnnModules.length) {
      val outputShapes = mklDnnModules(i).inferShape(shapes)
      require(outputShapes.length == 1, "submodule only output one tensor")
      outputShape.append(outputShapes(0))
    }
    outputShape.toArray
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(inputFormats()), "Memory formats should be inited")
    require(mklDnnModules != null, "You should call compile first")
    val buffer = new ArrayBuffer[MemoryData]()
    mklDnnModules.foreach(m => {
      m.initFwdPrimitives(runtime, phase)
      val out = m.outputFormats()
      require(out.length == 1, "output should be one tensor")
      buffer.append(out(0))
    })
    _outputFormats = buffer.toArray
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    val formats = gradOutputFormats()._1
    require(MemoryData.noUndef(formats), "Memory formats should be inited")
    val buffer = new ArrayBuffer[MemoryData]()
    mklDnnModules.foreach(m => {
      m.initBwdPrimitives(runtime, phase)
      val out = m.gradInputFormats()
      require(out.length == 1, "output should be one tensor")
      buffer.append(out(0))
    })
    _gradInputFormats = buffer.toArray
  }

  override private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    val formats = gradOutputFormats()._2
    require(MemoryData.noUndef(formats), "Memory formats should be inited")
    mklDnnModules.foreach(m => {
      m.initGradWPrimitives(runtime, phase)
    })
  }

  override private[mkldnn] def inputFormats() = {
    if (_inputFormats == null) {
      require(mklDnnModules != null, "container should be compiled")
      mklDnnModules.foreach { m =>
        require(m.inputFormats().length == 1, "input should be one tensor")
        if (_inputFormats == null) {
          _inputFormats = m.inputFormats()
        } else {
          require(_inputFormats(0) == m.inputFormats()(0), "input format should be same")
        }
      }
    }
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
    if (_gradOutputFormats == null) {
      require(mklDnnModules != null, "container should be compiled")
      val gradBuffer = new ArrayBuffer[MemoryData]()
      val gradForWeightBuffer = new ArrayBuffer[MemoryData]()
      mklDnnModules.foreach { m =>
        val (grad, gradForWeight) = m.gradOutputFormats()
        require(grad.length == 1, "module gradOutput should be one tensor")
        require(gradForWeight.length == 1, "module gradOutput should be one tensor")
        gradBuffer.append(grad(0))
        gradForWeightBuffer.append(gradForWeight(0))
      }
      _gradOutputFormats = (gradBuffer.toArray, gradForWeightBuffer.toArray)
    }
    _gradOutputFormats
  }

  override private[mkldnn] def initMemory() = {
    super.initMemory()
    gradInput = gradInputFormats()(0) match {
      case h: HeapData => Tensor[Float]()
      case n: NativeData => DnnTensor[Float](n.shape)
      case _ => throw new UnsupportedOperationException("NOt support memory format")
    }
  }

  private var _inputFormats: Array[MemoryData] = _
  private var _gradInputFormats: Array[MemoryData] = _
  private var _outputFormats: Array[MemoryData] = _
  private var _gradOutputFormats: (Array[MemoryData], Array[MemoryData]) = _
}

object ConcatTable {
  def apply(): ConcatTable = new ConcatTable()
}
