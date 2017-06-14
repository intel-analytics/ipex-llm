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

import com.google.protobuf.Extension.MessageType
import com.google.protobuf.{ExtensionRegistry, GeneratedMessage}

import scala.reflect.runtime.universe._
import com.intel.analytics.bigdl.nn.Abs
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import serialization.Model.{BigDLModel, CustomParam}
import serialization.Model.BigDLModel.ModuleType

import scala.collection.mutable
import scala.reflect.ClassTag

object CustomizedDelegator extends AbstractModelSerializer {
  private val customizedModules = new mutable.HashMap[Class[_], String]()
  private val customizedSerializers = new mutable.HashMap[String, AbstractModelSerializer]()
  private val registry = ExtensionRegistry.newInstance();
  override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
  : BigDLModule[T] = {
    val moduleType = model.getCustomParam.getCustomType
    if (customizedSerializers.contains(moduleType)) {
      customizedSerializers(moduleType).loadModule(model)
    } else {
      throw new IllegalArgumentException(s"$moduleType is not supported!")
    }
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]): BigDLModel = {
    var cls : Class[_] = null
    customizedModules.keySet.foreach(m => {
      if (m == module.module.getClass) {
        cls = m
      }
    })
    if (cls != null) {
      customizedSerializers(customizedModules(cls)).serializeModule(module)
    } else {
      throw new IllegalArgumentException(s"${module.module} is not supported")
    }
  }
  def registerCustomizedModule(cls : Class[_],
    serializer: AbstractModelSerializer,
    extension: GeneratedMessage.GeneratedExtension[_, _],
    customType : String): Unit = {
    require(!customizedSerializers.contains(customType), s"$customType already exist!")
    customizedModules(cls) = customType
    customizedSerializers(customType) = serializer
    registry.add(extension)
  }
  def getRegistry(): ExtensionRegistry = registry

  def getTypeTag[T : TypeTag](a : T) : TypeTag[T] = typeTag[T]
}
