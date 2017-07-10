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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * This class is a container for a single module which will be applied
 * to all input elements. The member module is cloned as necessary to
 * process all input elements.
 *
 * @param module
 */

@SerialVersionUID( 4403280698280280268L)
class MapTable[T: ClassTag](
  var module: AbstractModule[_ <: Activity, _ <: Activity, T] = null)
  (implicit ev: TensorNumeric[T]) extends Container[Table, Table, T]  {

  private def extend(n: Int): Unit = {
    modules.update(0, module.asInstanceOf[AbstractModule[Activity, Activity, T]])
    var i = 1
    while (i <= n && modules.size <= i) {
        modules.append(module
          .cloneModule()
          .asInstanceOf[AbstractModule[Activity, Activity, T]])
      i += 1
    }
  }

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(module != null, "Single module required")
    this.module = module
    if (modules.nonEmpty) {
      modules.update(0, module.asInstanceOf[AbstractModule[Activity, Activity, T]])
    } else {
      modules.append(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    this
  }

  override def updateOutput(input: Table): Table = {
    extend(input.length())
    var i = 0
    while (i < input.length()) {
      output.update(i + 1, modules(i).updateOutput(input(i + 1)))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    extend(input.length())
    var i = 0
    while (i < input.length()) {
      gradInput.update(i + 1, modules(i).updateGradInput(input(i + 1), gradOutput(i + 1)))
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    extend(input.length())
    var i = 0
    while (i < input.length()) {
        modules(i).accGradParameters(input(i + 1), gradOutput(i + 1))
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
    var str = s"${getPrintName}"
    if (module != null) {
      str += s"{$line$tab$module$line}"
    } else {
      str += " { }"
    }
    str
  }
}

object MapTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      module: AbstractModule[_ <: Activity, _ <: Activity, T] = null
  )(implicit ev: TensorNumeric[T]) : MapTable[T] = {
    new MapTable[T](module)
  }
}
