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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Sequential provides a means to plug layers together
 * in a feed-forward fully connected manner.
 */

@SerialVersionUID(5375403296928513267L)
class Sequential[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends DynamicContainer[Activity, Activity, T] {

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    var result = input.asInstanceOf[Activity]
    while (i < modules.length) {
      result = modules(i).forward(result)
      i += 1
    }

    this.output = result
    output
  }

  override def updateGradInput(input: Activity, nextError: Activity): Activity = {
    var i = modules.length - 1
    var error = nextError.asInstanceOf[Activity]
    while (i > 0) {
      val input = modules(i - 1).output
      error = modules(i).updateGradInput(input, error)
      i -= 1
    }
    error = modules(0).updateGradInput(input, error)

    this.gradInput = error
    gradInput
  }

  override def accGradParameters(
    input: Activity,
    gradOutput: Activity): Unit = {
    var i = modules.length - 1
    var currentModule = modules(i)
    var currentGradOutput = gradOutput
    while (i > 0) {
      val previousModule = modules(i - 1)
      currentModule.accGradParameters(previousModule.output, currentGradOutput)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
      i -= 1
    }

    currentModule.accGradParameters(input, currentGradOutput)
  }

  override def backward(input: Activity, nextError: Activity): Activity = {
    val before = System.nanoTime()
    var i = modules.length - 1
    var error = nextError.asInstanceOf[Activity]
    while (i > 0) {
      val input = modules(i - 1).output
      error = modules(i).backward(input, error)
      i -= 1
    }
    error = modules(0).backward(input, error)

    this.gradInput = error
    backwardTime += System.nanoTime() - before
    gradInput
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Sequential[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Sequential[T]]
    if (this.eq(other)) {
      return true
    }

    if (this.modules.length != other.modules.length) {
      return false
    }

    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      if (modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      hash = hash * seed + modules(i).hashCode()
      i += 1
    }

    hash
  }

  override def toString(): String = {
    val tab = "  "

    s"${getPrintName}{${line + tab}[input -> ${
      modules.zipWithIndex.map {
        case (m: AbstractModule[Activity, Activity, T], i: Int) => "(" + (i + 1) + ")"
      }.
        mkString(" -> ")
    } -> output]${line + tab}" +
      s"${
        modules.zipWithIndex.map {
          case (model: AbstractModule[Activity, Activity, T], index: Int)
          => s"(${index + 1}): ${model.setLine(line + tab)}"
        }.
          mkString(line + tab)
      }$line}"
  }

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    var startnodes = startNodes
    var curNodes: Array[ModuleNode[T]] = null
    for (i <- 0 to modules.size - 1) {
      curNodes = modules(i).getEndNodes(startnodes)
      startnodes = curNodes
    }
    curNodes
  }

  type ArrayBufferModules[T] = ArrayBuffer[AbstractModule[Activity, Activity, T]]
  private def doConvBn(modules: ArrayBufferModules[T]): this.type = {
    var i = 0
    while (i < modules.length) {
      val convMaybeIndex = i
      val bnMaybeIndex = i + 1
      val matched = bnMaybeIndex < modules.length &&
        modules(convMaybeIndex).isInstanceOf[mkldnn.ConvolutionDnn] &&
        modules(bnMaybeIndex).isInstanceOf[mkldnn.SpatialBatchNormalization[T]]

      if (matched) {
        // TODO transform weight and bias
        val bn = modules(bnMaybeIndex).asInstanceOf[mkldnn.SpatialBatchNormalization[T]]
        val conv = modules(convMaybeIndex).asInstanceOf[mkldnn.ConvolutionDnn]

        bn.runningVar.syncToHeap()
        bn.runningMean.syncToHeap()

        (0 until bn.nOutput).foreach { j =>
          val variance = bn.runningVar.storage().array()(j + bn.runningVar.storageOffset() - 1)
          val base = Math.sqrt(variance.asInstanceOf[Float] + bn.eps).toFloat

          val weight = if (conv.nGroup == 1) {
            conv.weight.select(1, j + 1)
          } else {
            conv.weight.select(2, j + 1)
          }
          weight.div(base)

          val bias = conv.bias.storage().array()(j)
          val mean = ev.toType[Float](bn.runningMean.storage().array()(i))
          conv.bias.storage().array()(j) = (bias - mean) / base
        }

        modules.remove(bnMaybeIndex)
      } else {
        modules(i).optimize()
        i += 1
      }
    }
    this
  }

  private def doConvRelu(modules: ArrayBufferModules[T]): this.type = {
    var i = 0
    while (i < modules.length) {
      val convMaybeIndex = i
      val reluMaybeIndex = i + 1
      val matched = reluMaybeIndex < modules.length &&
        modules(convMaybeIndex).isInstanceOf[mkldnn.ConvolutionDnn] &&
        modules(reluMaybeIndex).isInstanceOf[mkldnn.ReLUDnn[T]]

      if (matched) {
        modules(convMaybeIndex).asInstanceOf[mkldnn.ConvolutionDnn].setRelu(true)
        modules.remove(reluMaybeIndex)
      } else {
        modules(i).optimize()
        i += 1
      }
    }
    this
  }

  private def getLast(module: Module[T]): Module[T] = {
    module match {
      case _: Container[_, _, _] if module.isInstanceOf[Sequential[_]] =>
        module.asInstanceOf[Sequential[T]].modules.last
      case _ =>
        module
    }
  }

  private def doConvSum(modules: ArrayBufferModules[T]): this.type = {
    var i = 0
    val length = modules.length - 2
    while (i < length) {
      val concatTableMaybeIndex = i
      val caddTableMaybeIndex = i + 1
      val reluMaybeIndex = i + 2

      // check the two last elements of CAddTable.modules
      var shouldContinue = false
      shouldContinue = modules(concatTableMaybeIndex).isInstanceOf[mkldnn.ConcatTableDnn[T]] &&
        modules(caddTableMaybeIndex).isInstanceOf[mkldnn.CAddTableDnn[T]] &&
        modules(reluMaybeIndex).isInstanceOf[mkldnn.ReLUDnn[T]]

      var branch1: Module[T] = null
      var branch2: Module[T] = null
      if (shouldContinue) {
        val concatTable = modules(concatTableMaybeIndex).asInstanceOf[mkldnn.ConcatTableDnn[T]]

        if (concatTable.modules.length == 2) {
          branch1 = getLast(concatTable.modules(0))
          branch2 = getLast(concatTable.modules(1))

          def isConvOrIdentity(module: Module[T]): Boolean = {
            module.isInstanceOf[mkldnn.ConvolutionDnn] || module.isInstanceOf[Identity[T]]
          }

          shouldContinue = isConvOrIdentity(branch1) && isConvOrIdentity(branch2)

          // make sure the last module is conv
          if (!branch2.isInstanceOf[mkldnn.ConvolutionDnn]) {
            // swap the modules
            var tmp: Module[T] = null

            tmp = concatTable.modules(0)
            concatTable.modules(0) = concatTable.modules(1)
            concatTable.modules(1) = tmp

            tmp = branch1
            branch1 = branch2
            branch2 = tmp
          }
        }
      }

      if (shouldContinue) {
        // get the index of conv, by default the output should be the first conv.
        val (convIndex, conv, theOther) = (1, branch2.asInstanceOf[mkldnn.ConvolutionDnn], branch1)

        // delete CAddTable and ReLU
        modules.remove(caddTableMaybeIndex)
        modules.insert(caddTableMaybeIndex,
          mkldnn.DummyCAddTable[T](convIndex).asInstanceOf[Module[T]])
        modules.remove(reluMaybeIndex)
        // change the branch2's output to branch1's output
        conv.setSumOp(theOther.asInstanceOf[Module[Float]])
        conv.setSum(true)
      } else {
        modules(i).optimize()
        i += 1
      }
    }
    this
  }

  private def doBnRelu(modules: ArrayBufferModules[T]): this.type = {
    var i = 0
    while (i < modules.length) {
      val bnMaybeIndex = i
      val reluMaybeIndex = i + 1
      val matched = reluMaybeIndex < modules.length &&
        modules(bnMaybeIndex).isInstanceOf[mkldnn.SpatialBatchNormalization[T]] &&
        modules(reluMaybeIndex).isInstanceOf[mkldnn.ReLUDnn[T]]

      if (matched) {
        // TODO transform weight and bias
        modules(bnMaybeIndex).asInstanceOf[mkldnn.SpatialBatchNormalization[T]].relu = true
        modules.remove(reluMaybeIndex)
      } else {
        modules(i).optimize()
        i += 1
      }
    }
    this
  }

  override def optimize(): this.type = {
    if (modules.length >= 2) {
      doConvBn(modules).
        doConvRelu(modules).
        doBnRelu(modules).
        doConvSum(modules) // TODO
    }
    this
  }
}

object Sequential {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}
