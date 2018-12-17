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
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DeserializeContext, ModuleData, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule

import scala.collection.mutable.ArrayBuffer
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
  (implicit ev: TensorNumeric[T]) extends DynamicContainer[Table, Table, T]  {

  if ( module != null) {
    this.add(module)
  }

  private def extend(n: Int): Unit = {
    var i = 2
    while (i <= n && modules.size <= i) {
      if (modules.length <= i) {
        modules.append(module
          .cloneModule().setName(module.getName() + i)
          .asInstanceOf[AbstractModule[Activity, Activity, T]])
      }
      i += 1
    }
  }

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(module != null, "Single module required")
    this.module = module
    if (modules.nonEmpty) {
      modules.update(0, module.asInstanceOf[AbstractModule[Activity, Activity, T]])
      for (i <- 1 until modules.size) {
        modules.update(i, module.cloneModule().asInstanceOf[AbstractModule[Activity, Activity, T]])
      }
    } else {
      modules.append(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
    }
    this
  }

  override def updateOutput(input: Table): Table = {
    require(module != null, "Single module required")
    extend(input.length())
    var i = 0
    while (i < input.length()) {
      output.update(i + 1, modules(i).forward(input(i + 1)))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    require(module != null, "Single module required")
    extend(input.length())
    var i = 0
    while (i < input.length()) {
      gradInput.update(i + 1, modules(i).updateGradInput(input(i + 1), gradOutput(i + 1)))
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    require(module != null, "Single module required")
    extend(input.length())
    var i = 0
    while (i < input.length()) {
      modules(i).accGradParameters(input(i + 1), gradOutput(i + 1))
      i += 1
    }
  }

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    throw new IllegalArgumentException("Can not transform Container MapTable to graph")
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

  override def clearState(): this.type = {
    modules.clear()
    if ( module != null) {
      this.add(module)
    }
    this
  }
}

object MapTable extends ContainerSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
    module: AbstractModule[_ <: Activity, _ <: Activity, T] = null
  )(implicit ev: TensorNumeric[T]) : MapTable[T] = {
    new MapTable[T](module)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val mapTable = super.doLoadModule(context).asInstanceOf[MapTable[T]]
    require(mapTable.modules.size >=1, "sub module should not be empty")
    mapTable.add(mapTable.modules(0))
    mapTable
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              mapBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val mapTable = context.moduleData.module.asInstanceOf[MapTable[T]]
    val subModules = mapTable.modules
    require(subModules.size >=1, "sub module should not be empty")
    // `modules` are created during forward() by 'n' times of the same module depends on input size,
    // store the first one to save the storage cost just in case large input size
    val singleModule = subModules(0)
    mapTable.modules.clear()
    mapTable.modules.append(singleModule)
    super.doSerializeModule(context, mapBuilder)
  }
}
