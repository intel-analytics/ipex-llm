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
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * It is a container module that applies the i-th member module to the i-th
 * input, and outputs an output in the form of Table
 */

@SerialVersionUID(- 1197848941394786045L)
class ParallelTable[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends DynamicContainer[Table, Table, T] {

  override def updateOutput(input: Table): Table = {
    var i = 0
    while (i < input.length()) {
      output.update(i + 1, modules(i).forward(input(i + 1)))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    var i = 0
    while (i < input.length()) {
      gradInput.update(i + 1, modules(i).updateGradInput(input(i + 1), gradOutput(i + 1)))
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    var i = 0
    while (i < input.length()) {
      modules(i).accGradParameters(input(i + 1), gradOutput(i + 1))
      i += 1
    }
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    val before = System.nanoTime()
    var i = 0
    while (i < input.length()) {
      gradInput.update(i + 1, modules(i).backward(input(i + 1), gradOutput(i + 1)))
      i += 1
    }
    backwardTime += System.nanoTime() - before
    gradInput
  }

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    val outputs = ArrayBuffer[ModuleNode[T]]()
    var outputTuple: Array[ModuleNode[T]] = null
    require(startNodes.length == modules.length, s"ParallelTable: " +
      s"startNodes length ${startNodes.length} is more than modules length ${modules.length}")
    for (i <- 0 to modules.size - 1) {
      outputTuple = modules(i).getEndNodes(Array(startNodes(i)))
      outputs ++= outputTuple
    }
    outputs.toArray
  }

  override def toString: String = {
    val tab = "\t"
    val line = "\n"
    val next = "  |`-> "
    val lastNext = "   `-> "
    val ext = "  |    "
    val extlast = "       "
    val last = "   ... -> "
    var str = "nn.ParallelTable"
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

object ParallelTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : ParallelTable[T] = {
    new ParallelTable[T]()
  }
}
