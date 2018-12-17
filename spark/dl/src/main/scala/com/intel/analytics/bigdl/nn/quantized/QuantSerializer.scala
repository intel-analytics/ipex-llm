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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.TensorConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

trait QuantSerializer extends ModuleSerializable {
  def serializeWeight[T: ClassTag](context: SerializeContext[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit

  def serializeBias[T: ClassTag](context: SerializeContext[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val moduleData = context.moduleData
    val paramTable : Table = moduleData.module.getParametersTable()
    val moduleName = moduleData.module.getName()

    if (paramTable != null && paramTable.contains(moduleName)) {
      val modulePramTable: Table = paramTable(moduleName)
      val bias: Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias")
      } else {
        null
      }

      if (bias != null) {
        val biasAttr = AttrValue.newBuilder
        TensorConverter.setAttributeValue(context, biasAttr, bias)
        modelBuilder.setBias(biasAttr.getTensorValue)
      }
    }
  }

  def serializeOthers[T: ClassTag](context: SerializeContext[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
  }

  def loadWeight[T: ClassTag](context: DeserializeContext,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit

  def loadBias[T: ClassTag](context: DeserializeContext,
    moduleData: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val moduleName = moduleData.module.getName()
    val paramTable : Table = moduleData.module.getParametersTable
    if (paramTable != null && paramTable.contains(moduleName)) {
      val modulePramTable : Table = paramTable(moduleName)
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias")
      } else {
        null
      }

      if (bias != null) {
        val attrValue = AttrValue.newBuilder
        attrValue.setTensorValue(context.bigdlModule.getBias)
        val bias = TensorConverter.getAttributeValue(context, attrValue.build)
        modulePramTable("bias").asInstanceOf[Tensor[T]].copy(bias.asInstanceOf[Tensor[T]])
      }
    }
  }

  def loadOthers[T: ClassTag](context: DeserializeContext,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
  }

  override protected def copyFromBigDL[T: ClassTag](context: SerializeContext[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val storageType = context.storageType
    if (storageType == ProtoStorageType) {
      serializeWeight(context, modelBuilder)
      serializeBias(context, modelBuilder)
      serializeOthers(context, modelBuilder)
    } else {
      throw new IllegalArgumentException(s"$storageType not supported!")
    }
  }

  override protected def copy2BigDL[T: ClassTag](context: DeserializeContext, module: ModuleData[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    loadWeight(context, module)
    loadBias(context, module)
    loadOthers(context, module)
  }
}
