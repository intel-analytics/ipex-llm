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
import com.intel.analytics.bigdl.tensor.Tensor
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
class ConcatTable[T : ClassTag]
  (implicit ev: TensorNumeric[T])
  extends DynamicContainer[Activity, Table, T] with MklInt8Convertible {

  override def updateOutput(input: Activity): Table = {
    require(modules.length > 0, "empty modules of concat table")
    if (gradInput == null) {
      gradInput = allocateAs(input)
    }
    var i = 0
    while (i < modules.length) {
      val currentOutput = modules(i).forward(input)
      output.toTable(i + 1) = currentOutput
      i += 1
    }
    output
  }

  /**
   * add in to out
 *
   * @param out a table
   * @param in a table
   */
  private def addTable(out: Activity, in: Activity) : Unit = {
    if (in.isInstanceOf[Tensor[T]] && out.isInstanceOf[Tensor[T]]) {
      require(in.toTensor[T].nElement() == out.toTensor[T].nElement(),
        "gradInput should have the same size" +
          s"The sizes are ${in.toTensor[T].nElement()} " +
          s"and ${out.toTensor[T].nElement()}")
      out.toTensor[T].add(in.toTensor[T])
    } else {
      var i = 1
      while (i <= out.toTable.length()) {
        addTable(out.toTable(i), in.toTable(i))
        i += 1
      }
    }
  }

  /**
   * copy src to out
 *
   * @param out a table
   * @param src a table
   */
  private def copyTable(out: Activity, src: Activity) : Unit = {
    if (src.isInstanceOf[Tensor[T]] && out.isInstanceOf[Tensor[T]]) {
      out.toTensor[T].resizeAs(src.toTensor[T]).copy(src.toTensor[T])
    } else {
      var i = 1
      while (i <= out.toTable.length()) {
        copyTable(out.toTable(i), src.toTable(i))
        i += 1
      }
    }
  }

  /**
   * return a clone of src,
   * Notice: this is a deep copy, while Table.clone is a shallow copy.
 *
   * @param src a table
   * @return cloned table of src
   */
  private def cloneTable(src: Activity) : Activity = {
    if (src.isInstanceOf[Tensor[T]]) {
      src.toTensor[T].clone()
    } else {
      val out = T()
      var i = 1
      while (i <= src.toTable.length()) {
        out(i) = cloneTable(src.toTable(i))
        i += 1
      }
      out
    }
  }

  override def updateGradInput(input: Activity, gradOutput: Table): Activity = {
    require(modules.length > 0, "empty modules of concat table")
    val isInputTable = input.isInstanceOf[Table]
    val wasGradInputTable = gradInput.isInstanceOf[Table]

    if (isInputTable) {
      var i = 0
      while (i < modules.length) {
        val currentGradInput = modules(i).updateGradInput(input,
          gradOutput.toTable(i + 1))
        require(currentGradInput.isInstanceOf[Table],
          "currentGradInput is not a table!")
        if (i == 0) {
          if (!wasGradInputTable ||
            gradInput.toTable.length() != currentGradInput.toTable.length()) {
            // We need deep copy here.
            gradInput = cloneTable(currentGradInput)
          } else {
            copyTable(gradInput, currentGradInput)
          }
        } else {
          addTable(gradInput, currentGradInput)
        }
        i += 1
      }

    } else {
      var i = 0
      while (i < modules.length) {
        val currentGradInput = modules(i).updateGradInput(input,
          gradOutput.toTable(i + 1)).toTensor[T]
        if (i == 0) {
          if (wasGradInputTable) {
            gradInput = currentGradInput.clone()
          } else {
            gradInput.toTensor[T].resizeAs(
              currentGradInput).copy(currentGradInput)
          }
        } else {
          gradInput.toTensor[T].add(currentGradInput)
        }
        i += 1
      }
    }
    gradInput
  }

  override def backward(input: Activity, gradOutput: Table): Activity = {
    val before = System.nanoTime()
    require(modules.length > 0, "empty modules of concat table")
    val isInputTable = input.isInstanceOf[Table]
    val wasGradInputTable = gradInput.isInstanceOf[Table]

    if (isInputTable) {
      var i = 0
      while (i < modules.length) {
        val currentGradInput = modules(i).backward(input,
          gradOutput.toTable(i + 1))
        require(currentGradInput.isInstanceOf[Table],
          "currentGradInput is not a table!")
        if (i == 0) {
          if (!wasGradInputTable ||
            gradInput.toTable.length() != currentGradInput.toTable.length()) {
            // We need deep copy here.
            gradInput = cloneTable(currentGradInput)
          } else {
            copyTable(gradInput, currentGradInput)
          }
        } else {
          addTable(gradInput, currentGradInput)
        }
        i += 1
      }

    } else {
      var i = 0
      while (i < modules.length) {
        val currentGradInput = modules(i).backward(input,
          gradOutput.toTable(i + 1)).toTensor[T]
        if (i == 0) {
          if (wasGradInputTable) {
            gradInput = currentGradInput.clone()
          } else {
            gradInput.toTensor[T].resizeAs(
              currentGradInput).copy(currentGradInput)
          }
        } else {
          gradInput.toTensor[T].add(currentGradInput)
        }
        i += 1
      }
    }
    backwardTime += System.nanoTime() - before
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Table): Unit = {
    var i = 0
    while (i < modules.length) {
      modules(i).accGradParameters(input, gradOutput.toTable(i + 1))
      i += 1
    }
  }

  override def clearState(): ConcatTable.this.type = {
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

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    val outputs = ArrayBuffer[ModuleNode[T]]()
    var outputTuple: Array[ModuleNode[T]] = null
    for (i <- 0 to modules.size - 1) {
      outputTuple = modules(i).getEndNodes(startNodes)
      outputs ++= outputTuple
    }
    outputs.toArray
  }
}

object ConcatTable {
  def apply[A <: Activity : ClassTag, @specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : ConcatTable[T] = {
    new ConcatTable[T]()
  }
}
