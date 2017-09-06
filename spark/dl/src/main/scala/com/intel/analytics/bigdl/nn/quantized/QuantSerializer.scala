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
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData, ModuleSerializable}
import scala.reflect.ClassTag
import serialization.Bigdl.{AttrValue, BigDLModule, BigDLTensor}
import scala.reflect.runtime.universe

trait QuantSerializer extends ModuleSerializable {
  def serializeWeight[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit

  def serializeBias[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(module.module.getName)) {
      val modulePramTable: Table = paramTable(module.module.getName)
      val bias: Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias")
      } else {
        null
      }

      if (bias != null) {
        val biasTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(bias, biasTensorBuilder)
        modelBuilder.setBias(biasTensorBuilder.build)
      }
    }
  }

  def serializeOthers[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
  }

  def loadWeight[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit

  def loadBias[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(model.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias")
      } else {
        null
      }

      if (bias != null) {
        copy2BigDLTensor(bias, model.getBias)
      }
    }
  }

  def loadOthers[T: ClassTag](model: BigDLModule,
    module: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val quantModule = module.module.asInstanceOf[QuantizedModule[T]]
    quantModule.init()
  }

  override protected def copyFromBigDL[T: ClassTag](module: ModuleData[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    serializeWeight(module, modelBuilder)
    serializeBias(module, modelBuilder)
    serializeOthers(module, modelBuilder)
  }

  override protected def copy2BigDL[T: ClassTag](model: BigDLModule, module: ModuleData[T])
          (implicit ev: TensorNumeric[T]): Unit = {
    loadWeight(model, module)
    loadBias(model, module)
    loadOthers(model, module)
  }
}
