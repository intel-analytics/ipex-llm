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

import com.intel.analytics.bigdl.nn.Container

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.DataConverter.RegularizerConverter
import serialization.Bigdl.DataType
import serialization.Bigdl.{AttrValue, BigDLModule, BigDLTensor}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[ModuleSerializable]] trait inherits [[Loadable]] and [[Savable]]
 * traits for module serialization
 * it provides default implementation from [[ModuleSerializer]] using reflection
 */
trait ModuleSerializable extends Loadable with Savable{

  override def loadModule[T: ClassTag](model : BigDLModule)
                                      (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    ModuleSerializer.loadModule(model)
  }

  override def serializeModule[T: ClassTag](module : ModuleData[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModule = {
    ModuleSerializer.serializeModule(module)
  }

  protected def createBigDLModule[T: ClassTag](model : BigDLModule,
                                               module : AbstractModule[Activity, Activity, T])
                                              (implicit ev: TensorNumeric[T])
  : ModuleData[T] = {
    val preModules = model.getPreModulesList.asScala
    val nextModules = model.getNextModulesList.asScala
    val bigDLModule = ModuleData(module, preModules, nextModules)
    module.setName(model.getName)
    copy2BigDL(model, bigDLModule)
    bigDLModule
  }

  protected def createSerializeBigDLModule[T: ClassTag](
    modelBuilder : BigDLModule.Builder, module : ModuleData[T])(implicit ev: TensorNumeric[T])
  : BigDLModule = {
    module.pre.foreach(pre => modelBuilder.addPreModules(pre))
    module.next.foreach(next => modelBuilder.addNextModules(next))
    modelBuilder.setName(module.module.getName)
    copyFromBigDL(module, modelBuilder)
    modelBuilder.build
  }

  /**
   * copy serialized data (weight and bias if exist) to BigDL module
   * @param model serialized module
   * @param module  bigDL Module with relationships
   */
  protected def copy2BigDL[T: ClassTag](model : BigDLModule, module : ModuleData[T])
                                       (implicit ev: TensorNumeric[T]): Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(model.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val weight : Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight") }
      else null
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias") }
      else null
      if (weight != null) copy2BigDLTensor(weight, model.getWeight)
      if (bias != null) copy2BigDLTensor(bias, model.getBias)
    }
  }

  private def copy2BigDLTensor[T: ClassTag](tensor : Tensor[T], serializedTensor : BigDLTensor)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val dataType = serializedTensor.getDatatype
    if (dataType == DataType.FLOAT) {
      val serializedData = serializedTensor.getFloatDataList
      require(tensor.nElement() == serializedData.size(), "data size is not equal")
      var i = 0
      val tensorData = tensor.storage().array()
      var offset = tensor.storageOffset() - 1
      while (i < serializedData.size()) {
        tensorData(offset) = ev.fromType[Float](serializedData.get(i))
        offset += 1
        i += 1
      }
    } else if (dataType == DataType.DOUBLE) {
      val serializedData = serializedTensor.getDoubleDataList
      require(tensor.nElement() == serializedData.size(), "data size is not equal")
      var i = 0
      val tensorData = tensor.storage().array()
      var offset = tensor.storageOffset() - 1
      while (i < serializedData.size()) {
        tensorData(offset) = ev.fromType[Double](serializedData.get(i))
        offset += 1
        i += 1
      }
    }
  }

  /**
   * copy BigDL module data (weight and bias if exist) to BigDL Model to be persisted
   * @param modelBuilder serialized module builder
   * @param module  bigDL Module with relationships
   */
  protected def copyFromBigDL[T: ClassTag](module : ModuleData[T],
    modelBuilder : BigDLModule.Builder)(implicit ev : TensorNumeric[T]) : Unit = {
    val paramTable : Table = module.module.getParametersTable
    if (paramTable != null && paramTable.contains(module.module.getName)) {
      val modulePramTable : Table = paramTable(module.module.getName)
      val weight : Tensor[T] = if (modulePramTable.contains("weight")) {
        modulePramTable("weight") }
      else null
      val bias : Tensor[T] = if (modulePramTable.contains("bias")) {
        modulePramTable("bias") }
      else null
      if (weight != null) {
        val weightTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(weight, weightTensorBuilder)
        modelBuilder.setWeight(weightTensorBuilder.build)
      }
      if (bias != null) {
        val biasTensorBuilder = BigDLTensor.newBuilder
        copyFromBigDLTensor(bias, biasTensorBuilder)
        modelBuilder.setBias(biasTensorBuilder.build)
      }
    }
  }

  private def copyFromBigDLTensor[T: ClassTag](tensor : Tensor[T],
    serializedTensor : BigDLTensor.Builder)(implicit ev: TensorNumeric[T]) : Unit = {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
    val tensorData = tensor.storage().array()
    if (ev == NumericFloat) {
      var i = 0
      while (i < tensorData.length) {
        serializedTensor.addFloatData(ev.toType[Float](tensorData(i)))
        i += 1
      }
      serializedTensor.setDatatype(DataType.FLOAT)
    } else if (ev == NumericDouble) {
      var i = 0
      while (i < tensorData.length) {
        serializedTensor.addDoubleData(ev.toType[Double](tensorData(i)))
        i += 1
      }
      serializedTensor.setDatatype(DataType.DOUBLE)
    }
    tensor.size().foreach(_ => serializedTensor.addSize(_))
  }

}

trait ContainerSerializable extends ModuleSerializable {

  override def loadModule[T: ClassTag](model : BigDLModule)
                                      (implicit ev: TensorNumeric[T]) : ModuleData[T] = {
    val moduleData = ModuleSerializer.loadModule(model)
    val container = moduleData.module.asInstanceOf[Container[Activity, Activity, T]]
    val subModules = model.getSubModulesList.asScala
    subModules.foreach(module => {
      val subModuleData = ModuleSerializer.load(module)
      container.modules.append(subModuleData.module)
    })
    moduleData
  }

  override def serializeModule[T: ClassTag](module : ModuleData[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModule = {
    val bigDLModule = ModuleSerializer.serializeModule(module)
    val containerBuilder = BigDLModule.newBuilder(bigDLModule)
    val subModulesData = module.module.asInstanceOf[Container[Activity, Activity, T]].modules
    subModulesData.foreach(module => {
      val subModule = ModuleSerializer.serialize(ModuleData(module,
        new ArrayBuffer[String](), new ArrayBuffer[String]()))
      containerBuilder.addSubModules(subModule)
    })
    containerBuilder.build
  }
}

case class ModuleData[T: ClassTag](module : AbstractModule[Activity, Activity, T],
                                    pre : Seq[String], next : Seq[String])
trait Loadable {

  def loadModule[T: ClassTag](model : BigDLModule)
                             (implicit ev: TensorNumeric[T]) : ModuleData[T]
}
trait Savable {

  def serializeModule[T: ClassTag](module : ModuleData[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModule
}
