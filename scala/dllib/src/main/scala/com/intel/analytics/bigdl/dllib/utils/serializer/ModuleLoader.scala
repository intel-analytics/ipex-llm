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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.File
import serialization.Bigdl.BigDLModule

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object ModuleLoader {

  def loadFromFile[T: ClassTag](modelPath : String)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val modelBuilder = BigDLModule.newBuilder
    val inputBytes = File.readBytes(modelPath)
    var cis : CodedInputStream = CodedInputStream.newInstance(new ByteArrayInputStream(inputBytes))
    cis.setSizeLimit(Integer.MAX_VALUE)
    modelBuilder.mergeFrom(cis)
    val bigDLModel = modelBuilder.build()
    ModuleSerializer.load(bigDLModel).module
  }
}

object ModulePersister {

  def saveToFile[T: ClassTag](modelPath: String, module: AbstractModule[Activity, Activity, T],
                              overwrite: Boolean = false)(implicit ev: TensorNumeric[T]): Unit = {

    val bigDLModule = ModuleData(module
      , new ArrayBuffer[String](), new ArrayBuffer[String]())
    val bigDLModel = ModuleSerializer.serialize(bigDLModule)
    File.saveBytes(bigDLModel.toByteArray, modelPath, overwrite)
  }

  def saveModelDefinitionToFile[T: ClassTag](definitionPath : String,
    module : AbstractModule[Activity, Activity, T],
    overwrite : Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = ModuleData(module, new ArrayBuffer[String](), new ArrayBuffer[String]())
    val bigDLModel = ModuleSerializer.serialize(bigDLModule)
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
