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
package com.intel.analytics.bigdl.utils

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream, FileOutputStream}

import com.google.protobuf.CodedInputStream
import com.intel.analytics.bigdl.nn.{Linear, Sequential}

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IOUtils
import serialization.Model.BigDLModel.ModuleType
import serialization.Model.{BigDLModel, BigDLTensor}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

sealed abstract class ModelSerializer {

  def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T]) : BigDLModule[T]

  def serializeModule[T: ClassTag](module : BigDLModule[T])
                                  (implicit ev: TensorNumeric[T]) : BigDLModel
  def copy2BigDL[T: ClassTag](model : BigDLModel, module : BigDLModule[T])
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

  def copyFromBigDL[T: ClassTag](module : BigDLModule[T], modelBuilder : BigDLModel.Builder)
                      (implicit ev : TensorNumeric[T]) : Unit = {
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
                               tops : ArrayBuffer[String], bottoms : ArrayBuffer[String])
object ModelSerializer {
  private val serializerMap = new mutable.HashMap[String, ModelSerializer]()
  private val hdfsPrefix: String = "hdfs:"
  serializerMap("LINEAR") = LinearSerializer
  serializerMap("SEQUENTIAL") = SequentialSerializer
  case object LinearSerializer extends ModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
      : BigDLModule[T] = {
      val bottomList = model.getBottomsList.asScala
      val topList = model.getTopsList.asScala
      val params = model.getModuleParamsList.asScala
      val name = model.getName
      val inputSize = params(0).toInt
      val outputSize = params(1).toInt
      val linear = Linear[T](inputSize, outputSize).setName(name)
      var tops = new ArrayBuffer[String]()
      val bottoms = new ArrayBuffer[String]()
      topList.foreach(top => tops.append(top))
      bottomList.foreach(bottom => bottoms.append(bottom))
      val bigDLModule = BigDLModule(linear, tops,
        bottoms)
      copy2BigDL(model, bigDLModule)
      bigDLModule
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
      (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      module.bottoms.foreach(_ => bigDLModelBuilder.addAllBottoms(_))
      module.tops.foreach(_ => bigDLModelBuilder.addTops(_))
      bigDLModelBuilder.setName(module.module.getName)
      copyFromBigDL(module, bigDLModelBuilder)
      val linear = module.module.asInstanceOf[Linear[T]]
      bigDLModelBuilder.addModuleParams(linear.inputSize)
      bigDLModelBuilder.addModuleParams(linear.outputSize)
      bigDLModelBuilder.setModuleType(ModuleType.LINEAR)
      bigDLModelBuilder.build
    }
  }
  case object SequentialSerializer extends ModelSerializer {
    override def loadModule[T: ClassTag](model : BigDLModel)(implicit ev: TensorNumeric[T])
    : BigDLModule[T] = {
      val bottomList = model.getBottomsList.asScala
      val topList = model.getTopsList.asScala
      val subModules = model.getSubModulesList.asScala
      val name = model.getName
      val sequantial = Sequential[T]().setName(name)
      subModules.foreach(subModule => {
        val bigDLModule = load(subModule)
        sequantial.add(bigDLModule.module.
          asInstanceOf[AbstractModule[_ <: Activity, _ <: Activity, T]])
      })
      var tops = new ArrayBuffer[String]()
      val bottoms = new ArrayBuffer[String]()
      topList.foreach(top => tops.append(top))
      bottomList.foreach(bottom => bottoms.append(bottom))
      val bigDLModule = BigDLModule(sequantial, tops, bottoms)
      copy2BigDL(model, bigDLModule)
      bigDLModule
    }
    override def serializeModule[T: ClassTag](module : BigDLModule[T])
      (implicit ev: TensorNumeric[T]): BigDLModel = {
      val bigDLModelBuilder = BigDLModel.newBuilder
      module.bottoms.foreach(_ => bigDLModelBuilder.addAllBottoms(_))
      module.tops.foreach(_ => bigDLModelBuilder.addTops(_))
      bigDLModelBuilder.setName(module.module.getName)
      copyFromBigDL(module, bigDLModelBuilder)
      val sequential = module.module.asInstanceOf[Sequential[T]]
      sequential.modules.foreach(subModule => {
        val subModel = serialize(BigDLModule(subModule,
          new ArrayBuffer[String](), new ArrayBuffer[String]()))
        bigDLModelBuilder.addSubModules(subModel)
      })
      bigDLModelBuilder.setModuleType(ModuleType.SEQUENTIAL)
      bigDLModelBuilder.build
    }
  }
  private def load[T: ClassTag](model: BigDLModel)
    (implicit ev: TensorNumeric[T]) : BigDLModule[T] = {
    serializerMap(model.getModuleType.toString).loadModule(model)
  }
  private def serialize[T: ClassTag](bigDLModule : BigDLModule[T])
    (implicit ev: TensorNumeric[T])
    : BigDLModel = {
    // serializerMap(module.getModuleType.toString).loadModule(model)
    val module = bigDLModule.module.asInstanceOf[AbstractModule[_, _, _]]
    val bigDLModel = module match {
      case linear : Linear[_] => LinearSerializer.serializeModule(bigDLModule)
      case sequantial : Sequential[_] => SequentialSerializer.serializeModule(bigDLModule)
    }
    bigDLModel
  }

  def loadFromFile[T: ClassTag](modelPath : String)
    (implicit ev: TensorNumeric[T]) : AbstractModule[_, _, _] = {
    val modelBuilder = BigDLModel.newBuilder
    var cis : CodedInputStream = null
    if (modelPath.startsWith(hdfsPrefix)) {
      val byteArrayOut = com.intel.analytics.bigdl.utils.File.readHdfsByte(modelPath)
      cis = CodedInputStream.newInstance(new ByteArrayInputStream(byteArrayOut))
    } else {
      cis = CodedInputStream.newInstance(new FileInputStream(modelPath))
    }
    cis.setSizeLimit(Integer.MAX_VALUE)
    modelBuilder.mergeFrom(cis)
    val bigDLModel = modelBuilder.build()
    load(bigDLModel).module
  }

  def saveToFile[T: ClassTag](modelPath : String, bigDLModule : BigDLModule[T],
    overwrite: Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModel = serialize(bigDLModule)
    if (modelPath.startsWith(hdfsPrefix)) {
      val binaryFile = new Path(modelPath)
      val fs = binaryFile.getFileSystem(new Configuration())
      if (fs.exists(binaryFile)) {
        if (overwrite) {
          fs.delete(binaryFile, true)
        } else {
          throw new RuntimeException(s"file $modelPath already exists")
        }
      }
      val out = fs.create(binaryFile)
      val byteArrayOut = new ByteArrayOutputStream()
      byteArrayOut.write(bigDLModel.toByteArray)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } else {
      val binaryFile = new java.io.File(modelPath)
      if (binaryFile.exists()) {
        if (overwrite) {
          binaryFile.delete()
        } else {
          throw new RuntimeException(s"file $modelPath already exists")
        }
      }
      val binaryWriter = new FileOutputStream(binaryFile)
      binaryWriter.write(bigDLModel.toByteArray)
      binaryWriter.close
    }
  }
}
