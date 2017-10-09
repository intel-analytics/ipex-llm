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

import java.io._

import scala.collection.JavaConverters._
import com.google.protobuf.CodedInputStream
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, Table}
import serialization.Bigdl.{BigDLModule, DataType, TensorStorage}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object ModuleLoader {

  /**
   * load module from `modelPath`
   * @param modelPath  path where protobuf formatted module is stored
   * @param ev numeric ops
   * @tparam T data type
   * @return loaded BigDL module
   */
  def loadFromFile[T: ClassTag](modelPath : String)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val modelBuilder = BigDLModule.newBuilder
    val inputBytes = File.readBytes(modelPath)
    var cis : CodedInputStream = CodedInputStream.newInstance(new ByteArrayInputStream(inputBytes))
    cis.setSizeLimit(Integer.MAX_VALUE)
    modelBuilder.mergeFrom(cis)
    val bigDLModel = modelBuilder.build()
    val storages = new mutable.HashMap[Int, Any]()
    // loadAllStorages(bigDLModel, storages)
    ModuleSerializer.load(DeserializeContext(bigDLModel, storages, ProtoStorageType)).module
  }

  /**
   * Load weights from `modulePath` and copy to pre-defined module
   * for `layers` layers, copy all if not specified
   * @param definition  pre-defined module
   * @param modelPath   path where protobuf formatted module is stored
   * @param layers  name list of layers weight & bias of which to be copied
   * @param ev  numeric ops
   * @tparam T data type
   */

  def loadFromDefinition[T : ClassTag](definition : AbstractModule[Activity, Activity, T],
    modelPath : String, layers : mutable.HashSet[String] = null)(implicit ev: TensorNumeric[T])
  : Unit = {
    val loadedModule = loadFromFile(modelPath)
    val layersToCopy = if (layers == null) {
      val allLayers = new mutable.HashSet[String]()
      getAllLayers(definition, allLayers)
      allLayers
    } else {
      layers
    }
    copyParams(definition, loadedModule, layersToCopy)
  }

  private def getAllLayers[T : ClassTag](module : AbstractModule[Activity, Activity, T],
    layers : mutable.HashSet[String]) : Unit
    = {
    layers.add(module.getName)
    if (module.isInstanceOf[Container[_, _, _]]) {
      module.asInstanceOf[Container[_, _, _]].modules.foreach(subModule => {
        getAllLayers(subModule, layers)
      })
    }
  }

  private def copyParams[T : ClassTag](definition : AbstractModule[Activity, Activity, T],
                                     mirror : AbstractModule[Activity, Activity, T],
                                      layers : mutable.HashSet[String]) : Unit = {
    val parameterTable = definition.getParametersTable()
    val copiedParameterTable = mirror.getParametersTable()
    layers.foreach(name => {
      if (parameterTable.contains(name)) {
        require(copiedParameterTable.contains(name), s"$name does not exist in loaded module")
        copyParams(parameterTable.get(name).get.asInstanceOf[Table],
          copiedParameterTable.get(name).get.asInstanceOf[Table])
      }
    })
  }

  private def copyParams[T : ClassTag](params : Table, copyParams : Table) : Unit = {
    copyParam(params, copyParams, "weight")
    copyParam(params, copyParams, "bias")
  }

  private def copyParam[T : ClassTag](params : Table,
                                      copyParams : Table, paraName : String) : Unit = {
    if (params.contains(paraName)) {
      // this is for quantization tensors where the weight might be an array
      if (copyParams.get(paraName).get
        .isInstanceOf[Array[Tensor[T]]]) {
        require(params.get(paraName).get
          .isInstanceOf[Array[Tensor[T]]], "param type mismatch!")
        val copies = params.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        val origins = params.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        var i = 0
        while (i < copies.length) {
          origins(i).copy(copies(i))
          i += 1
        }
      } else {
        // For normal layers, their params are just tensors
        params.get(paraName).get.asInstanceOf[Tensor[T]].copy(
          copyParams.get(paraName).get.asInstanceOf[Tensor[T]])
      }
    }
  }
}

object ModulePersister {

  /**
   * Persist module to specified path
   * @param modelPath path to persist module to
   * @param module  module to be persisted
   * @param overwrite if overwrite module file if exists
   * @param ev  numeric ops
   * @tparam T data type
   */
  def saveToFile[T: ClassTag](modelPath: String, module: AbstractModule[Activity, Activity, T],
                              overwrite: Boolean = false)
                             (implicit ev: TensorNumeric[T]): Unit = {
    val bigDLModule = ModuleData(module
      , new ArrayBuffer[String](), new ArrayBuffer[String]())
    val storages = new mutable.HashMap[Int, Any]()
    val context = SerializeContext(bigDLModule, storages, ProtoStorageType)
    val bigDLModel = ModuleSerializer.serialize(context).bigDLModule
    File.saveBytes(bigDLModel.toByteArray, modelPath, overwrite)
  }

  /**
   * Save module definition to given path
   * @param definitionPath the path to persist definition path to
   * @param module  module to be persisted
   * @param overwrite if overwrite module file if exists
   * @param ev numeric ops
   * @tparam T data type
   */
  def saveModelDefinitionToFile[T: ClassTag](definitionPath : String,
    module : AbstractModule[Activity, Activity, T],
    overwrite : Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = ModuleData(module, new ArrayBuffer[String](), new ArrayBuffer[String]())
    val storages = new mutable.HashMap[Int, Any]()
    val context = SerializeContext(bigDLModule, storages, ProtoStorageType)
    val bigDLModel = ModuleSerializer.serialize(context).bigDLModule
    val bigDLModelWithoutWeightsAndBias = BigDLModule.newBuilder(bigDLModel)
    cleantWeightAndBias(bigDLModelWithoutWeightsAndBias)
    val model = bigDLModelWithoutWeightsAndBias.build
    val byteArrayOut = new ByteArrayOutputStream()
    byteArrayOut.write(model.toString.getBytes)
    File.saveBytes(byteArrayOut.toByteArray, definitionPath, overwrite)
  }

  private def cleantWeightAndBias(modelBuilder : BigDLModule.Builder): Unit = {
    modelBuilder.clearWeight
    modelBuilder.clearBias
    if (modelBuilder.getSubModulesCount > 0) {
      val subModules = modelBuilder.getSubModulesList
      modelBuilder.clearSubModules
      subModules.asScala.foreach(sub => {
        val subModelBuilder = BigDLModule.newBuilder(sub)
        cleantWeightAndBias(subModelBuilder)
        modelBuilder.addSubModules(subModelBuilder.build)
      })
    }
  }
}
