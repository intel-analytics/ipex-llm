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
package com.intel.analytics.bigdl.utils.serializer

import scala.collection.JavaConverters._
import scala.reflect.runtime.universe
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Model.BigDLModel

import scala.collection.mutable
import scala.reflect.ClassTag


trait ModuleSerializable extends Loadable with Savable{

  override def loadModule[T: ClassTag](model : BigDLModel)
                                       (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    ModuleSerializer.loadModule(model)
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModel = {
    ModuleSerializer.serializeModule(module)
  }

}

trait Loadable {

  def loadModule[T: ClassTag](model : BigDLModel)
                             (implicit ev: TensorNumeric[T]) : BigDLModule[T]
}
trait Savable {

  def serializeModule[T: ClassTag](module : BigDLModule[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModel
}

object ModuleSerializer extends ModuleSerializable{

  override def loadModule[T: ClassTag](model : BigDLModel)
                                      (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {

    val modelAttributes = model.getAttrMap
    val moduleType = model.getModuleType
    val preModules = model.getPreModulesList.asScala
    val nextModules = model.getNextModulesList.asScala
    val cls = ModuleSerializer.getModuleClsByType(moduleType)

    BigDLModule(null, preModules, nextModules)
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModel = {
    null
  }


  val moduleMaps = new mutable.HashMap[String, Class[_]]()

  def registerModule(moduleType : String, cls : Class[_]) : Unit = {
    moduleMaps(moduleType) = cls
  }

  def getModuleClsByType(moduleType : String) : Class[_] = {
    require(moduleMaps.contains(moduleType), s"$moduleType is not supported")
    moduleMaps(moduleType)
  }
  init
  def init() : Unit = {
    registerModule("ABS", Class.forName("com.intel.analytics.bigdl.nn.Abs"))
    registerModule("ADDCONSTANT", Class.forName("com.intel.analytics.bigdl.nn.AddConstant"))
  }
}

case class BigDLModule[T: ClassTag](module : AbstractModule[Activity, Activity, T],
                                    pre : Seq[String], next : Seq[String])