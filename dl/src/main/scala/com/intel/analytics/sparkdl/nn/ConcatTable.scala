/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{Activities, T, Table}

import scala.reflect.ClassTag

class ConcatTable[T : ClassTag](implicit ev: TensorNumeric[T])
  extends Container[Activities, Activities, T] {

  output = T()

  override def updateOutput(input: Activities): Activities = {
    var i = 0
    while (i < modules.length) {
      val currentOutput = modules(i).updateOutput(input)
      if (!output.toTable().contains(i + 1)) {
        output.toTable().insert(i + 1, currentOutput)
      } else if (currentOutput != output.toTable().get(i + 1).get) {
        output.toTable().update(i + 1, currentOutput)
      }
      i += 1
    }
    output
  }

  /**
   * add in to out
   * @param out
   * @param in
   */
  private def addTable(out: Activities, in: Activities) : Unit = {
    if (in.isInstanceOf[Tensor[T]] && out.isInstanceOf[Tensor[T]]) {
      require(in.toTensor[T]().nElement() == out.toTensor[T]().nElement(),
        "gradInput should have the same size")
      out.toTensor[T]().add(in.toTensor[T]())
    } else {
      var i = 1
      while (i <= out.toTable().length()) {
        addTable(out.toTable().get[Activities](i).get, in.toTable().get[Activities](i).get)
        i += 1
      }
    }
  }

  /**
   * copy in to out
   * @param out
   * @param in
   */
  private def copyTable(out: Activities, in: Activities) : Unit = {
    if (in.isInstanceOf[Tensor[T]] && out.isInstanceOf[Tensor[T]]) {
      out.toTensor[T]().resizeAs(in.toTensor[T]()).copy(in.toTensor[T]())
    } else {
      var i = 1
      while (i <= out.toTable().length()) {
        copyTable(out.toTable().get[Activities](i).get, in.toTable().get[Activities]().get)
        i += 1
      }
    }
  }

  /**
   * return a clone of in
   * @param in
   * @return cloned table
   */
  private def cloneTable(in: Activities) : Activities = {
    if (in.isInstanceOf[Tensor[T]]) {
      in.toTensor[T]().clone()
    } else {
      val out = T()
      var i = 1
      while (i <= in.toTable().length()) {
        out(i) = cloneTable(in.toTable()(i))
        i += 1
      }
      out
    }
  }

  def backward(method: String, input: Activities, gradOutput: Activities,
    scale : Double = 1.0) : Activities = {

    val isTable = input.isInstanceOf[Table]
    val wasTable = gradInput.isInstanceOf[Table]

    if (isTable) {
      if (!wasTable) {
        gradInput = null
      }
      var i = 0
      while (i < modules.length) {
        method match {
          case "updateGradInput" =>
            val currentGradInput = modules(i).updateGradInput(input,
              gradOutput.toTable().get(i + 1).get)
            require(currentGradInput.isInstanceOf[Table],
              "currentGradInput is not a table!")
            if (i == 0) {
              if (null == gradInput) {
                gradInput = cloneTable(currentGradInput)
              } else {
                copyTable(gradInput, currentGradInput)
              }
            } else {
              addTable(gradInput, currentGradInput)
            }
          case "accGradParameters" =>
            modules(i).accGradParameters(input, gradOutput.toTable().get(i + 1).get, scale)
        }
        i += 1
      }

    } else {
      if (wasTable) {
        gradInput = null
      }
      var i = 0
      while (i < modules.length) {
        method match {
          case "updateGradInput" =>
            val currentGradInput = modules(i).updateGradInput(input,
              gradOutput.toTable().get(i + 1).get)
            if (i == 0) {
              if (null == gradInput) {
                gradInput = currentGradInput.toTensor().clone()
              } else {
                gradInput.toTensor[T]().resizeAs(
                  currentGradInput.toTensor[T]()).copy(currentGradInput.toTensor[T]())
              }
            } else {
              gradInput.toTensor[T]().add(currentGradInput.toTensor[T]())
            }
          case "accGradParameters" =>
            modules(i).accGradParameters(input, gradOutput.toTable().get(i + 1).get, scale)
        }
        i += 1
      }
    }
    gradInput
  }

  override def updateGradInput(input: Activities, gradOutput: Activities): Activities = {
    backward("updateGradInput", input, gradOutput)
  }

  override def accGradParameters(input: Activities, gradOutput: Activities,
    scale: Double = 0.1): Unit = {

    backward("accGradParameters", input, gradOutput)
  }

  override def toString(): String = {
    val tab = "\t"
    val line = "\n"
    val next = "  |`-> "
    val lastNext = "   `-> "
    val ext = "  |    "
    val extlast = "       "
    val last = "   ... -> "
    var str = "nn.ConcatTable"
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
