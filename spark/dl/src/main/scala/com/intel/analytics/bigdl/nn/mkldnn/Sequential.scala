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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{Sequential => Seq}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

class Sequential extends MklDnnContainer {

  val fuseConvBn = System.getProperty("bigdl.mkldnn.fusion.convbn", "false").toBoolean
  val fuseBnRelu = System.getProperty("bigdl.mkldnn.fusion.bnrelu", "false").toBoolean
  val fuseConvRelu = System.getProperty("bigdl.mkldnn.fusion.convrelu", "false").toBoolean
  val fuseConvSum = System.getProperty("bigdl.mkldnn.fusion.convsum", "false").toBoolean

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, Float]): this.type = {
    require(mklDnnModules == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  override private[mkldnn] def fusion(phase: Phase): Unit = {
    modules.clear()
    modules.appendAll(getFusedModules(phase).map {x =>
      x.asInstanceOf[AbstractModule[Activity, Activity, Float]]
    })
    mklDnnModules = modules.map(_.asInstanceOf[MklDnnModule]).toArray
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    var lastOutputFormats = inputs
    var firstRealInputFormats: Array[MemoryData] = null
    for (i <- 0 until mklDnnModules.length) {
      val m = mklDnnModules(i)
      val (realInputFormats, outputFormats) = m.initFwdPrimitives(lastOutputFormats, phase)
      lastOutputFormats.zip(realInputFormats).foreach {
        case (o, i) => reorderManager.register(o, i)
      }
      if (i == 0) firstRealInputFormats = realInputFormats
      lastOutputFormats = outputFormats
    }
    (firstRealInputFormats, lastOutputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    var lastGradInputFormats = grads
    var firstRealGradOutputFormats: Array[MemoryData] = null
    for (i <- mklDnnModules.length - 1 to 0 by -1) {
      val m = mklDnnModules(i)
      val (realGradOutput, gradInputFomrats) = m.initBwdPrimitives(lastGradInputFormats, phase)
      lastGradInputFormats.zip(realGradOutput).foreach {
        case (gi, go) => reorderManager.register(gi, go)
      }
      if (i == mklDnnModules.length - 1) firstRealGradOutputFormats = realGradOutput
      lastGradInputFormats = gradInputFomrats
    }
    (firstRealGradOutputFormats, lastGradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grads: Array[MemoryData], phase: Phase) = {
    var lastGradInputFormats = grads
    var firstRealGradOutputFormats: Array[MemoryData] = null
    for (i <- mklDnnModules.length - 1 to 0 by -1) {
      val m = mklDnnModules(i)
      val realGradOutput = m.initGradWPrimitives(lastGradInputFormats, phase)
      lastGradInputFormats.zip(realGradOutput).foreach {
        case (gi, go2) => reorderManager.register(gi, go2)
      }
      if (i == mklDnnModules.length - 1) firstRealGradOutputFormats = realGradOutput
      lastGradInputFormats = m.gradInputFormats()
    }
    firstRealGradOutputFormats
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
        mklDnnModules(i - 1).gradOutputFormats(),
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
      currentModule = modules(i)
      val curInput = reorderManager.infer(
        mklDnnModules(i - 1).outputFormats(),
        mklDnnModules(i).inputFormats(),
        modules(i - 1).output
      )
      currentModule.accGradParameters(curInput, lastGradInput)
      lastGradInput = reorderManager.infer(
        mklDnnModules(i).gradInputFormats(),
        mklDnnModules(i - 1).gradOutputWeightFormats(),
        modules(i).gradInput
      )
      i -= 1
    }

    modules(i).accGradParameters(input, lastGradInput)
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

  override private[mkldnn] def gradOutputWeightFormats() = {
    modules.last.asInstanceOf[MklDnnModule].gradOutputWeightFormats()
  }

  type ArrayBufferModules[Float] = ArrayBuffer[AbstractModule[Activity, Activity, Float]]
  private def convWithBn(modules: Array[MklDnnModule], phase: Phase): Array[MklDnnModule] = {
    if (fuseConvBn && phase == InferencePhase) {
      val newModules: ArrayBuffer[MklDnnModule] = ArrayBuffer.empty
      var lastBn: SpatialBatchNormalization = null

      modules.zip(modules.drop(1) ++ Array(null)).foreach { case (f, s) =>
        (f, s) match {
          case (conv: SpatialConvolution, bn: SpatialBatchNormalization) =>
            mergeConvBn(conv, bn)
            newModules.append(conv)
            lastBn = bn
          case (f: MklDnnContainer, s) => f.fusion(phase); newModules.append(f)
          case _ => if (lastBn != f) { newModules.append(f) }
        }
      }

      newModules.toArray
    } else {
      modules
    }
  }

  private def convWithReLU(modules: Array[MklDnnModule], phase: Phase): Array[MklDnnModule] = {
    if (fuseConvRelu) {
      val newModules: ArrayBuffer[MklDnnModule] = ArrayBuffer.empty
      var lastReLU: ReLU = null

      modules.zip(modules.drop(1) ++ Array(null)).foreach { case (f, s) =>
        (f, s) match {
          case (conv: SpatialConvolution, relu: ReLU) =>
            newModules.append(conv)
            conv.setReLU()
            lastReLU = relu
          case (f: MklDnnContainer, s) =>
            f.fusion(phase)
            newModules.append(f)
          case _ => if (lastReLU != f) {
            newModules.append(f)
          }
        }
      }

      newModules.toArray
    } else {
      modules
    }
  }

  private def bnWithReLU(modules: Array[MklDnnModule], phase: Phase): Array[MklDnnModule] = {
    if (fuseBnRelu) {
      val newModules: ArrayBuffer[MklDnnModule] = ArrayBuffer.empty
      var lastReLU: ReLU = null

      modules.zip(modules.drop(1) ++ Array(null)).foreach { case (f, s) =>
        (f, s) match {
          case (bn: SpatialBatchNormalization, relu: ReLU) =>
            newModules.append(bn)
            bn.setReLU(true)
            lastReLU = relu
          case (f: MklDnnContainer, s) => f.fusion(phase); newModules.append(f)
          case _ => if (lastReLU != f) { newModules.append(f) }
        }
      }

      newModules.toArray
    } else {
      modules
    }
  }

  private def convWithSum(modules: Array[MklDnnModule], phase: Phase): Array[MklDnnModule] = {
    val newModules: ArrayBuffer[MklDnnModule] = ArrayBuffer.empty
    if (!fuseConvSum || modules.length <= 2 || phase == TrainingPhase) {
      newModules.appendAll(modules)
    } else {
      var lastConv: SpatialConvolution = null
      var lastReLU: ReLU = null

      modules.zip(modules.drop(1) ++ Array(null)).foreach {
        case (f: ConcatTable, s: CAddTable) => val (conv, sbt) = convSum(f, s)
          newModules.append(f)
          lastConv = conv
          if (sbt != null) {
            newModules.append(sbt)
          }
        case (f: MklDnnContainer, s) => f.fusion(phase); newModules.append(f)
        case (f: CAddTable, s: ReLU) => if (lastConv != null) {
          lastConv.setReLU()
          lastReLU = s
          lastConv = null
        } else {
          newModules.append(f)
        }
        case (f, s) => if (lastReLU != f) { newModules.append(f); lastReLU = null}
      }
    }

    newModules.toArray
  }

  private def getFusedModules(phase: Phase): Array[MklDnnModule] = {
    val f1Modules = convWithBn(mklDnnModules, phase)
    val f2Modules = convWithReLU(f1Modules, phase)
    val f3Modules = bnWithReLU(f2Modules, phase)
    val f4Modules = convWithSum(f3Modules, phase)
    f4Modules
  }

  private def mergeConvBn(conv: SpatialConvolution, bn: SpatialBatchNormalization): Unit = {

    val originVar = Tensor[Float].resize(bn.runningVariance.size()).copy(bn.runningVariance.dense)
    val originMean = Tensor[Float].resize(bn.runningMean.size()).copy(bn.runningMean.dense)

    val convWeight = Tensor[Float].resize(conv.weight.size()).copy(conv.weight.dense)
    val convBias = Tensor[Float].resize(conv.bias.size()).copy(conv.bias.dense)

    (0 until bn.nOutput).foreach { j =>
      val variance = originVar.storage().array()(j + originVar.storageOffset() - 1)
      val base = Math.sqrt(variance.asInstanceOf[Float] + bn.eps).toFloat
      require(base != 0.0, s"the eps of ${bn.getName()} should be more than 0")

      val weight = if (conv.nGroup == 1) {
        convWeight.select(1, j + 1)
      } else {
        convWeight.select(2, j + 1)
      }
      weight.div(base)

      val bias = convBias.storage().array()(j)
      val mean = originMean.storage().array()(j)
      convBias.storage().array()(j) = (bias - mean) / base
    }

    conv.weight.copy(convWeight)
    conv.bias.copy(convBias)
  }

  private def getLast(
    module: AbstractModule[Activity, Activity, Float]): AbstractModule[Activity, Activity, Any] = {
    val ret = module match {
      case sequential: Sequential => sequential.modules.last
      case _ => module
    }

    ret.asInstanceOf[AbstractModule[Activity, Activity, Any]]
  }

  private def convSum(concatTable: ConcatTable, cAddTable: CAddTable): (SpatialConvolution,
    SelectTable) = {
    var branch1: AbstractModule[Activity, Activity, Any] = null
    var branch2: AbstractModule[Activity, Activity, Any] = null

    var continue = concatTable.modules.length == 2

    if (continue) {
      branch1 = getLast(concatTable.modules(0))
      branch2 = getLast(concatTable.modules(1))

      def isConvOrIdentity(module: AbstractModule[Activity, Activity, Any]): Boolean = {
        module.isInstanceOf[SpatialConvolution] || module.isInstanceOf[Identity]
      }

      continue = continue && isConvOrIdentity(branch1) && isConvOrIdentity(branch2)
    }

    if (continue) {
      // make sure the last module is conv
      if (!branch2.isInstanceOf[SpatialConvolution]) {
        // swap the modules
        var tmp: AbstractModule[Activity, Activity, Float] = null

        tmp = concatTable.modules(0)
        concatTable.modules(0) = concatTable.modules(1)
        concatTable.modules(1) = tmp

        tmp = branch1.asInstanceOf[AbstractModule[Activity, Activity, Float]]
        branch1 = branch2
        branch2 = tmp.asInstanceOf[AbstractModule[Activity, Activity, Any]]
      }

      // get the index of conv, by default the output should be the first conv.
      val (convIndex, conv, theOther) = (1, branch2.asInstanceOf[SpatialConvolution], branch1)
      conv.setSum()

      // delete CAddTable
      val selectTable = SelectTable(convIndex)

      // change the branch2's output to branch1's output
      // FIXME maybe we should not set the conv operation
      conv.setSumOp(theOther.asInstanceOf[Module[Float]])
      (conv, selectTable)
    } else {
      (null, null)
    }
  }

  override def toString(): String = {
    val tab = "  "

    s"${getPrintName}{${line + tab}[input -> ${
      modules.zipWithIndex.map {
        case (m: AbstractModule[Activity, Activity, Float], i: Int) => "(" + (i + 1) + ")"
      }.
        mkString(" -> ")
    } -> output]${line + tab}" +
      s"${
        modules.zipWithIndex.map {
          case (model: AbstractModule[Activity, Activity, Float], index: Int)
          => s"(${index + 1}): ${model.setLine(line + tab)}"
        }.
          mkString(line + tab)
      }$line}"
  }
}

object Sequential {
  def apply(): Sequential = new Sequential()
}
