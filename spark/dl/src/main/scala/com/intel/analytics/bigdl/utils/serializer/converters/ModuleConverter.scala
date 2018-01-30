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

package com.intel.analytics.bigdl.utils.serializer.converters

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, DataType}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe


/**
 * DataConverter for [[com.intel.analytics.bigdl.nn.abstractnn.AbstractModule]]
 */
object ModuleConverter extends DataConverter {

  override def getAttributeValue[T: ClassTag](context: DeserializeContext, attribute: AttrValue)
                                             (implicit ev: TensorNumeric[T]): AnyRef = {
    val serializedModule = attribute.getBigDLModuleValue
    if (serializedModule.getModuleType != null && serializedModule.getModuleType != "") {
      ModuleSerializer.load(DeserializeContext(serializedModule,
        context.storages, context.storageType)).module
    } else {
      null
    }
  }

  override def setAttributeValue[T: ClassTag](context: SerializeContext[T],
    attributeBuilder: AttrValue.Builder,
    value: Any, valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
    attributeBuilder.setDataType(DataType.MODULE)
    if (value != null) {
      val module = value.asInstanceOf[AbstractModule[Activity, Activity, T]]
      val serializableModule = ModuleSerializer.
        serialize(SerializeContext(ModuleData(module, Seq[String](), Seq[String]()),
          context.storages, context.storageType)).bigDLModule
      attributeBuilder.setBigDLModuleValue(serializableModule)
    }
  }
}

