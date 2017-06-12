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
package com.intel.analytics.bigdl.utils.serialization

import scala.collection.JavaConverters._

import com.intel.analytics.bigdl.nn.{BilinearFiller, Default, InitializationMethod, Xavier}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{L1L2Regularizer, L1Regularizer, L2Regularizer, Regularizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import serialization.Model
import serialization.Model.{BigDLModel, BigDLTensor, RegularizerType}

import scala.reflect.ClassTag


private[serialization] abstract class AbstractModelSerializer {

  def loadModule[T: ClassTag](model : BigDLModel)
    (implicit ev: TensorNumeric[T]) : BigDLModule[T]

  def serializeModule[T: ClassTag](module : BigDLModule[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModel

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

  protected def createBigDLModule[T: ClassTag](model : BigDLModel,
    module : AbstractModule[Activity, Activity, T])
      (implicit ev: TensorNumeric[T])
  : BigDLModule[T] = {
    val tops = model.getTopsList.asScala
    val bottoms = model.getBottomsList.asScala
    val bigDLModule = BigDLModule(module, tops, bottoms)
    module.setName(model.getName)
    copy2BigDL(model, bigDLModule)
    bigDLModule
  }

  protected def createSerializeBigDLModule[T: ClassTag](
    modelBuilder : BigDLModel.Builder, module : BigDLModule[T])(implicit ev: TensorNumeric[T])
  : BigDLModel = {
    module.bottoms.foreach(bottom => modelBuilder.addBottoms(bottom))
    module.tops.foreach(top => modelBuilder.addTops(top))
    modelBuilder.setName(module.module.getName)
    copyFromBigDL(module, modelBuilder)
    modelBuilder.build
  }

  protected def createRegularizer[T: ClassTag]
  (modelRegularizer : Model.Regularizer)(implicit ev: TensorNumeric[T]): Regularizer[T] = {
    modelRegularizer.getRegularizerType match {
      case RegularizerType.L1Regularizer => L1Regularizer(modelRegularizer.getRegularData(0))
      case RegularizerType.L2Regularizer => L2Regularizer(modelRegularizer.getRegularData(0))
      case RegularizerType.L1L2Regularizer => L1L2Regularizer(modelRegularizer.getRegularData(0),
        modelRegularizer.getRegularData(1))
      case _ => throw new IllegalArgumentException(s"${modelRegularizer.getRegularizerType}" +
        s"cannot be recognized")
    }
  }

  protected def createSerializeRegularizer[T: ClassTag]
  (regularizer : Regularizer[T])(implicit ev: TensorNumeric[T]): Model.Regularizer = {
    val builder = Model.Regularizer.newBuilder
    regularizer match {
      case reg : L1Regularizer[T] =>
        builder.setRegularizerType(RegularizerType.L1Regularizer)
        builder.addRegularData(regularizer.asInstanceOf[L1Regularizer[T]].l1)
      case reg : L2Regularizer[T] =>
        builder.setRegularizerType(RegularizerType.L2Regularizer)
        builder.addRegularData(regularizer.asInstanceOf[L2Regularizer[T]].l2)
      case reg : L1L2Regularizer[T] =>
        builder.setRegularizerType(RegularizerType.L1L2Regularizer)
        val l1l2 = regularizer.asInstanceOf[L1L2Regularizer[T]]
        builder.addRegularData(l1l2.l1)
        builder.addRegularData(l1l2.l2)
    }
    builder.build
  }

  protected def createInitMethod[T: ClassTag]
  (initMethod : Model.InitMethod)(implicit ev: TensorNumeric[T]): InitializationMethod = {
    initMethod match {
      case Model.InitMethod.Default => Default
      case Model.InitMethod.Xavier => Xavier
      case Model.InitMethod.BilinearFiller => BilinearFiller
      case _ => throw new IllegalArgumentException(s"${initMethod}" +
        s"cannot be recognized")
    }
  }

  protected def createSerializeInitMethod[T: ClassTag]
  (initMethod : InitializationMethod)(implicit ev: TensorNumeric[T]): Model.InitMethod = {
    initMethod match {
      case Default => Model.InitMethod.Default
      case Xavier => Model.InitMethod.Xavier
      case  BilinearFiller => Model.InitMethod.BilinearFiller
      case _ => throw new IllegalArgumentException(s"${initMethod}" +
        s"cannot be recognized")
    }
  }
}

private[serialization] case class BigDLModule[T: ClassTag](
  module : AbstractModule[Activity, Activity, T],
  tops : Seq[String], bottoms : Seq[String])