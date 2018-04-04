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

import com.intel.analytics.bigdl.nn.{ConcatTable, Container, DynamicContainer}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * ConcateTable is a container module like Concate. Applies an input
 * to each member module, input can be a tensor or a table.
 *
 * ConcateTable usually works with CAddTable and CMulTable to
 * implement element wise add/multiply on outputs of two modules.
 */
@SerialVersionUID(- 704681653938468956L)
class ConcatTableDnn[T : ClassTag]
(implicit ev: TensorNumeric[T]) extends DynamicContainer[Tensor[T], Table, T] {
  override def updateOutput(input: Tensor[T]): Table = {
    val s1 = System.nanoTime()
    require(modules.length > 0, "empty modules of concat table")
    var i = 0
    // todo: not care about format ???
    while (i < modules.length) {
      val currentOutput = modules(i).forward(input)
      output.toTable(i + 1) = currentOutput
      i += 1
    }
    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugFwInfo(this.getName(), end1, input.getFormat(), -2)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    val s1 = System.nanoTime()
    require(modules.length > 0, "empty modules of concat table")
    var format: Int = 0
    var i = 0
    while (i < modules.length) {
      val currentGradInput = modules(i).
                              updateGradInput(input, gradOutput.toTable(i + 1)).toTensor[T]
      if (i == 0) {
        if (gradInput.nElement() == 0) {
          gradInput = MklDnnTensor[T](currentGradInput.size())
        }
        gradInput.resize(currentGradInput.size())
        gradInput.copy(currentGradInput)
        format = currentGradInput.getFormat()
      } else {
        // todo: before add, need to transfor to same format, here just require
//        require(currentGradInput.getFormat() == format,
//                                      "all tensor results from modules should have same" +
//         s"format ${format} ${currentGradInput.getFormat()} ${modules(i).getName()} ${this.getName()}")

        gradInput.add(currentGradInput)
      }
      i += 1
    }
    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, format, gradInput.getFormat())
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Table): Unit = {
    var i = 0
    while (i < modules.length) {
      modules(i).accGradParameters(input, gradOutput.toTable(i + 1))
      i += 1
    }
  }

  override def clearState(): ConcatTableDnn.this.type = {
    super.clearState()
    modules.foreach(_.clearState())
    if (gradInput.isInstanceOf[Table]) {
      gradInput.toTable.clear()
    }
    this
  }

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

object ConcatTableDnn {
  def apply[A <: Activity : ClassTag, @specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : ConcatTableDnn[T] = {
    new ConcatTableDnn[T]()
  }
}
