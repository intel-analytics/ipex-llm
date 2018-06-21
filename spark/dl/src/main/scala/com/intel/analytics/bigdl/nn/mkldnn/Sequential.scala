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

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{DynamicContainer, Sequential => Seq}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Sequential extends MklDnnContainer {

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, Float]): this.type = {
    require(mklDnnModules == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  override private[mkldnn] def fusion(phase: Phase): Unit = {
    modules.filter(_.isInstanceOf[MklDnnContainer])
      .map { case mc: MklDnnContainer => mc.fusion(phase) }
  }

  override private[mkldnn] def inferShape(shapes: Array[Array[Int]]) = {
    var lastShape = shapes
    modules.foreach { case m: MklDnnModule => lastShape = m.inferShape(lastShape)}
    lastShape
  }

  override private[mkldnn] def initFwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(mklDnnModules(0).inputFormats()), "Memory formats should be inited")
    mklDnnModules(0).initFwdPrimitives(runtime, phase)
    var lastOutputFormats = mklDnnModules(0).outputFormats()
    for (i <- 1 until mklDnnModules.length) {
      val m = mklDnnModules(i)
      lastOutputFormats.zip(m.inputFormats()).foreach {
        case (o, i) => if (i.layout == Memory.Format.format_undef) {
          i.setLayout(o.layout)
        }
      }
      m.initFwdPrimitives(runtime, phase)
      lastOutputFormats.zip(m.inputFormats()).foreach {
        case (o, i) => reorderManager.register(o, i, runtime, phase)
      }
      lastOutputFormats = m.outputFormats()
    }
  }

  override private[mkldnn] def inputFormats() = {
    modules(0).asInstanceOf[MklDnnModule].inputFormats()
  }

  override private[mkldnn] def gradInputFormats() = {
    modules(0).asInstanceOf[MklDnnModule].gradInputFormats()
  }

  override private[mkldnn] def outputFormats() = {
    modules.last.asInstanceOf[MklDnnModule].outputFormats()
  }

  override private[mkldnn] def gradOutputFormats() = {
    modules.last.asInstanceOf[MklDnnModule].gradOutputFormats()
  }

  override private[mkldnn] def initMemory() = {
    modules.foreach { case m: MklDnnModule => m.initMemory() }
  }

  override private[mkldnn] def initBwdPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(mklDnnModules.last.gradOutputFormats()._1),
      "Memory formats should be inited")
    mklDnnModules.last.initBwdPrimitives(runtime, phase)
    var lastGradInputFormats = mklDnnModules.last.gradInputFormats()
    for (i <- mklDnnModules.length - 2 to 0 by -1) {
      val m = mklDnnModules(i)
      lastGradInputFormats.zip(m.gradOutputFormats()._1).foreach {
        case (gi, go) => if (go.layout == Memory.Format.format_undef) {
          go.setLayout(gi.layout)
        }
      }
      m.initBwdPrimitives(runtime, phase)
      lastGradInputFormats.zip(m.gradOutputFormats()._1).foreach {
        case (gi, go) => reorderManager.register(gi, go, runtime, phase)
      }
      lastGradInputFormats = m.gradOutputFormats()._1
    }
  }

  override private[mkldnn] def initGradWPrimitives(runtime: MklDnnRuntime, phase: Phase) = {
    require(MemoryData.noUndef(mklDnnModules.last.gradOutputFormats()._2),
      "Memory formats should be inited")
    mklDnnModules.last.initGradWPrimitives(runtime, phase)
    var lastGradInputFormats = mklDnnModules.last.gradInputFormats()
    for (i <- mklDnnModules.length - 2 to 0 by -1) {
      val m = mklDnnModules(i)
      lastGradInputFormats.zip(m.gradOutputFormats()._2).foreach {
        case (gi, go2) => if (go2.layout == Memory.Format.format_undef) {
          go2.setLayout(gi.layout)
        }
      }
      m.initGradWPrimitives(runtime, phase)
      lastGradInputFormats.zip(m.gradOutputFormats()._2).foreach {
        case (gi, go2) => reorderManager.register(gi, go2, runtime, phase)
      }
      lastGradInputFormats = m.gradOutputFormats()._2
    }
  }

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    var lastOutput = input
    while (i < mklDnnModules.length - 1) {
      lastOutput = reorderManager.infer(
        mklDnnModules(i).outputFormats(),
        mklDnnModules(i + 1).inputFormats(),
        modules(i).forward(lastOutput)
      )
      i += 1
    }

    this.output = modules(i).forward(lastOutput)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    var i = modules.length - 1
    var lastGradInput = gradOutput
    while (i > 0) {
      val curInput = reorderManager.infer(
        mklDnnModules(i - 1).outputFormats(),
        mklDnnModules(i).inputFormats(),
        modules(i - 1).output
      )
      lastGradInput = reorderManager.infer(
        mklDnnModules(i).gradInputFormats(),
        mklDnnModules(i - 1).gradOutputFormats()._1,
        modules(i).updateGradInput(curInput, lastGradInput)
      )
      i -= 1
    }
    lastGradInput = modules(0).updateGradInput(input, lastGradInput)

    this.gradInput = lastGradInput
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = modules.length - 1
    var currentModule = modules(i)
    var lastGradInput = gradOutput
    while (i > 0) {
      val curInput = reorderManager.infer(
        mklDnnModules(i - 1).outputFormats(),
        mklDnnModules(i).inputFormats(),
        modules(i - 1).output
      )
      currentModule.accGradParameters(curInput, lastGradInput)
      lastGradInput = reorderManager.infer(
        mklDnnModules(i).gradInputFormats(),
        mklDnnModules(i - 1).gradOutputFormats()._2,
        modules(i).gradInput
      )
      i -= 1
    }

    currentModule.accGradParameters(input, lastGradInput)
  }
}

object Sequential {
  def apply(): Sequential = new Sequential()
}
