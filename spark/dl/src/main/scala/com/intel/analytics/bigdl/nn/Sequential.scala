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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Sequential provides a means to plug layers together
 * in a feed-forward fully connected manner.
 */

@SerialVersionUID(5375403296928513267L)
class Sequential[T: ClassTag]
(implicit ev: TensorNumeric[T])
  extends DynamicContainer[Activity, Activity, T] with MklInt8Convertible {

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
}

object Sequential {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}
