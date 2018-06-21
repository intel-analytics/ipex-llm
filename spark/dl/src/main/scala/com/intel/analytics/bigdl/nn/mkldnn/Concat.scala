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

import com.intel.analytics.bigdl.nn.DynamicContainer
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

class Concat(val dimension: Int) extends MklDnnContainer {
  private var _inputFormats: Array[MemoryData] = _
  private var _gradInputFormats: Array[MemoryData] = _
  private var _outputFormats: Array[MemoryData] = _
  private var _gradOutputFormats: (Array[MemoryData], Array[MemoryData]) = _
  private var _outputShape: Array[Int] = _

  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]) = {
    require(shapes.length == 1, "Concat only accept one tensor")
    mklDnnModules = modules.map(_.asInstanceOf[MklDnnModule]).toArray
    require(mklDnnModules.length > 0, "Concat should contains at least one module")

    for(i <- 0 until mklDnnModules.length) {
      val outputShapes = mklDnnModules(i).inferShape(shapes)
      require(outputShapes.length == 1, "submodule only output one tensor")
      if (_outputShape == null) {
        _outputShape = outputShapes(0)
      } else {
        require(_outputShape.length == outputShapes(0).length, "shape length doesn't match")
        for(i <- 0 until _outputShape.length) {
          if (i == dimension - 1) {
            _outputShape(i) += outputShapes(0)(i)
          } else {
            require(_outputShape(i) == outputShapes(0)(i), "shape doesn't match")
          }
        }
      }
    }
    require(dimension > 1 && dimension <= _outputShape.length, "invalid dimension")
    Array(_outputShape)
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(inputFormats()), "Memory formats should be inited")
    require(_outputShape != null, "You should call infer shape first")
    mklDnnModules.foreach(m => {
      m.initFwdPrimitives(runtime, phase)
      val out = m.outputFormats()
      require(out.length == 1, "output should be one tensor")
      if (_outputFormats == null) {
        _outputFormats = out
        // the expect input layout maybe auto
        _inputFormats(0).setLayout(m.inputFormats()(0).layout)
      } else {
        require(_outputFormats(0).layout == out(0).layout, "output layout not match")
        require(_inputFormats(0).layout == m.inputFormats()(0).layout, "input layout not match")
      }
    })
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(gradOutputFormats()._1), "Memory formats should be inited")
    mklDnnModules.foreach(m => {
      m.initBwdPrimitives(runtime, phase)
      val format = m.gradInputFormats()
      require(format.length == 1, "gradInput should be one tensor")
      if (_gradInputFormats == null) {
        _gradInputFormats = format
      } else {
        require(_gradInputFormats(0) == format(0), "gradInput memory format not match")
      }
    })
  }

  override private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(gradOutputFormats()._2), "Memory formats should be inited")
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
      require(_outputShape != null, "You should call infer shape first")
      var grad: MemoryData = null
      var gradForWeight: MemoryData = null
      mklDnnModules.foreach { m =>
        val moduleGradOutput = m.gradOutputFormats()
        require(moduleGradOutput._1 == 1, "gradOutput should be one tensor")
        require(moduleGradOutput._2 == 1, "gradOutput should be one tensor")
        if (grad == null) {
          grad = moduleGradOutput._1(0)
          gradForWeight = moduleGradOutput._2(0)
        } else {
          grad.setShape(moduleGradOutput._1(0).shape)
          require(grad == moduleGradOutput._1(0), "gradOutput format should be same")
          gradForWeight.setShape(moduleGradOutput._2(0).shape)
          require(gradForWeight == moduleGradOutput._2(0), "gradOutput format should be same")
        }
      }
      grad.setShape(_outputShape)
      gradForWeight.setShape(_outputShape)
      _gradOutputFormats = (Array(grad), Array(gradForWeight))
    }
    _gradOutputFormats
  }

  override def updateOutput(input: Activity): Activity = ???

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = ???
}
