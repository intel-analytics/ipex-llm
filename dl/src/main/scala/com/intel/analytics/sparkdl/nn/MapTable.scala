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

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{Activities, T, Table}

import scala.reflect.ClassTag

class MapTable[T: ClassTag](
  var module: Module[_ <: Activities, _ <: Activities, T] = null)
  (implicit ev: TensorNumeric[T]) extends Container[Table, Table, T]  {

  def extend(n: Int): Unit = {
    modules.update(0, module.asInstanceOf[Module[Activities, Activities, T]])
    var i = 1
    while (i <= n) {
      if (modules.size <= i) {
        modules.append(module
          .cloneModule()
          .asInstanceOf[Module[Activities, Activities, T]])
      }
      i += 1
    }
  }

  override def add(module: Module[_ <: Activities, _ <: Activities, T]): this.type = {
    require(module != null, "Single module required")
    this.module = module
    if (modules.nonEmpty) {
      modules.update(0, module.asInstanceOf[Module[Activities, Activities, T]])
    } else {
      modules.append(module.asInstanceOf[Module[Activities, Activities, T]])
    }
    this
  }

  override def updateOutput(input: Table): Table = {
    extend(input.getState().size)
    var i = 0
    while (i < input.getState().size) {
      output.update(i + 1, modules(i).updateOutput(input(i + 1)))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    extend(input.getState().size)
    var i = 0
    while (i < input.getState().size) {
      gradInput.update(i + 1, modules(i).updateGradInput(input(i + 1), gradOutput(i + 1)))
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table,
    scale: Double = 1.0): Unit = {
    extend(input.getState().size)
    var i = 0
    while (i < input.getState().size) {
        modules(i).accGradParameters(input(i + 1), gradOutput(i + 1), scale)
      i += 1
    }
  }


  override def zeroGradParameters(): Unit = {
    if (module != null) {
      module.zeroGradParameters()
    }
  }


  override def updateParameters(learningRate: T): Unit = {
    if (module != null) {
      module.updateParameters(learningRate)
    }
  }

  override def toString(): String = {
    val tab = "  "
    val extlast = "       "
    val line = "\n"
    var str = "nn.MapTable"
    if (module != null) {
      str += s"{$line$tab$module$line}"
    } else {
      str += " { }"
    }
    str
  }
}
