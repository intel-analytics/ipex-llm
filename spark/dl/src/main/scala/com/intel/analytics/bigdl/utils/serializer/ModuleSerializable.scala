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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.DataConverter.RegularizerConverter
import serialization.Model.AttrValue.DataType
import serialization.Model.{AttrValue, BigDLModel, BigDLTensor}

import scala.reflect.ClassTag

/**
 * [[ModuleSerializable]] trait inherits [[Loadable]] and [[Savable]]
 * traits for module serialization
 * it provides default implementation from [[ModuleSerializer]] using reflection
 */
trait ModuleSerializable extends Loadable with Savable{

  override def loadModule[T: ClassTag](model : BigDLModel)
                                      (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    ModuleSerializer.loadModule(model)
  }

  override def serializeModule[T: ClassTag](module : BigDLModule[T])
                                           (implicit ev: TensorNumeric[T]) : BigDLModel = {
    ModuleSerializer.serializeModule(module)
  }

  protected def createBigDLModule[T: ClassTag](model : BigDLModel,
                                               module : AbstractModule[Activity, Activity, T])
                                              (implicit ev: TensorNumeric[T])
  : BigDLModule[T] = {
    val preModules = model.getPreModulesList.asScala
    val nextModules = model.getNextModulesList.asScala
    val bigDLModule = BigDLModule(module, preModules, nextModules)
    module.setName(model.getName)
    copy2BigDL(model, bigDLModule)
    bigDLModule
  }

  protected def createSerializeBigDLModule[T: ClassTag](
    modelBuilder : BigDLModel.Builder, module : BigDLModule[T])(implicit ev: TensorNumeric[T])
  : BigDLModel = {
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
  protected def copy2BigDL[T: ClassTag](model : BigDLModel, module : BigDLModule[T])
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
    val serializedData = serializedTensor.getDataList
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

  /**
   * copy BigDL module data (weight and bias if exist) to BigDL Model to be persisted
   * @param modelBuilder serialized module builder
   * @param module  bigDL Module with relationships
   */
  protected def copyFromBigDL[T: ClassTag](module : BigDLModule[T],
    modelBuilder : BigDLModel.Builder)(implicit ev : TensorNumeric[T]) : Unit = {
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
    var i = 0
    val tensorData = tensor.storage().array()
    var offset = tensor.storageOffset() - 1
    while (i < tensorData.length) {
      serializedTensor.addData(ev.toType[Double](tensorData(i)))
      i += 1
    }
    tensor.size().foreach(_ => serializedTensor.addSize(_))
  }

}
case class BigDLModule[T: ClassTag](module : AbstractModule[Activity, Activity, T],
                                    pre : Seq[String], next : Seq[String])
trait Loadable {

  def loadModule[T: ClassTag](model : BigDLModel)
                             (implicit ev: TensorNumeric[T]) : BigDLModule[T]
}
trait Savable {

  def serializeModule[T: ClassTag](module : BigDLModule[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModel
}