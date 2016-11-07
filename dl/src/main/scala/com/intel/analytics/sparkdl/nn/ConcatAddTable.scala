
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
import com.intel.analytics.sparkdl.utils.Activities

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class ConcatAddTable[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends Container[Activities, Activities, T] {
  val concatOutput = ArrayBuffer[Tensor[T]]()
  val cAddGradInput = ArrayBuffer[Tensor[T]]()

  output = Tensor[T]()
  gradInput = Tensor[T]()
  override def updateOutput(input: Activities): Tensor[T] ={
    concatTableUpdateOutput(input.asInstanceOf[Tensor[T]])
    cADDTableUpdateOutput(concatOutput)
  }

  def concatTableUpdateOutput(input: Tensor[T]): ArrayBuffer[Tensor[T]] = {
    for ((module, i) <- modules.zipWithIndex) {
//      val tmpModel = module.asInstanceOf[TensorModule[T]]
      updateBuffer(i, concatOutput, module.updateOutput(input).asInstanceOf[Tensor[T]])
    }
    concatOutput
  }

  def cADDTableUpdateOutput(input: ArrayBuffer[Tensor[T]]): Tensor[T] = {
    if (ip) {
      output.asInstanceOf[Tensor[T]].set(input(0))
    } else {
      output.asInstanceOf[Tensor[T]].resizeAs(input(0)).copy(input(0))
    }

    var i = 1
    while (i < input.length) {
      output.asInstanceOf[Tensor[T]].add(input(i))
      i += 1
    }

    output.asInstanceOf[Tensor[T]]
  }


  def concatTableUpdateGradInput(input: Tensor[T], gradOutputs: ArrayBuffer[Tensor[T]]): Tensor[T] = {
    for ((module, i) <- modules.zipWithIndex) {
     // val tmpModel = module.asInstanceOf[TensorModule[T]]
      val currentGradOutput = gradOutputs(i)
      val currentGradInput = module.updateGradInput(input, currentGradOutput).asInstanceOf[Tensor[T]]
      if (i == 0) {
        gradInput.asInstanceOf[Tensor[T]].resizeAs(currentGradInput).copy(currentGradInput)
      } else {
        gradInput.asInstanceOf[Tensor[T]].add(currentGradInput)
      }
    }

    gradInput.asInstanceOf[Tensor[T]]
  }

  def updateBuffer(i: Int, arrayBuffer: ArrayBuffer[Tensor[T]], tensor: Tensor[T]) = {
    if (i < arrayBuffer.size)
      arrayBuffer.update(i, tensor)
    else
      arrayBuffer.append(tensor)
  }

  def cAddTableUpdateGradInput(input: ArrayBuffer[Tensor[T]], gradOutputs: Tensor[T]): ArrayBuffer[Tensor[T]] = {
    for ((in, i) <- input.zipWithIndex) {
      if (ip)
        updateBuffer(i, cAddGradInput, gradOutputs)
      else
        updateBuffer(i, cAddGradInput, Tensor().resizeAs(in).copy(gradOutputs))
    }

    cAddGradInput
  }

  override def updateGradInput(input: Activities, gradOutput: Activities): Tensor[T] = {
    cAddTableUpdateGradInput(concatOutput, gradOutput.asInstanceOf[Tensor[T]])
    concatTableUpdateGradInput(input.asInstanceOf[Tensor[T]], cAddGradInput)
  }

  override def accGradParameters(input: Activities, gradOutput: Activities, scale: Double): Unit = {
    for ((module, i) <- modules.zipWithIndex) {
      module.accGradParameters(input, cAddGradInput(i), scale)
    }
  }

  override def toString: String = {
    val tab = "  "
    val line = "\n"
    val next = "  |`-> "
    val ext = "  |    "
    val extLast = "       "
    val last = "   ... -> "
    s"nn.ConcatAddTable {$line${tab}input$line${
      modules.zipWithIndex
        .map { case (model: Module[Activities, Activities, T], index: Int) =>
          s"$tab$next(${index + 1}): ${
            if (index == modules.length - 1) {
              model.setLine(line + tab + extLast)
            } else {
              model.setLine(line + tab + ext)
            }
          }"
        }.mkString(line)
    }$line$tab${last}output$line$tab}"
  }
}