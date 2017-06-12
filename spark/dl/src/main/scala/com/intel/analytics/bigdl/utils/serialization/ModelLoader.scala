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

import java.io._
import scala.collection.JavaConverters._

import com.google.protobuf.CodedInputStream
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IOUtils
import serialization.Model.BigDLModel

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object ModelLoader {

  private val hdfsPrefix: String = "hdfs:"

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
    ModelSerializer.load(bigDLModel).module
  }
}

object ModelPersister {

  private val hdfsPrefix: String = "hdfs:"

  def saveToFile[T: ClassTag](modelPath : String, module : AbstractModule[_, _, T],
                              overwrite: Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = BigDLModule(module.asInstanceOf[AbstractModule[Activity, Activity, T]]
      , new ArrayBuffer[String](), new ArrayBuffer[String]())
    val bigDLModel = ModelSerializer.serialize(bigDLModule)
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

  def saveModelDefinitionToFile[T: ClassTag](definitionPath : String,
    module : AbstractModule[Activity, Activity, T],
    overwrite : Boolean = false)(implicit ev: TensorNumeric[T]) : Unit = {
    val bigDLModule = BigDLModule(module, new ArrayBuffer[String](), new ArrayBuffer[String]())
    val bigDLModel = ModelSerializer.serialize(bigDLModule)
    val bigDLModelWithoutWeightsAndBias = BigDLModel.newBuilder(bigDLModel)
    cleantWeightAndBias(bigDLModelWithoutWeightsAndBias)
    val model = bigDLModelWithoutWeightsAndBias.build
    if (definitionPath.startsWith(hdfsPrefix)) {
      val prototxtFile = new Path(definitionPath)
      val fs = prototxtFile.getFileSystem(new Configuration())
      if (fs.exists(prototxtFile)) {
        if (overwrite) {
          fs.delete(prototxtFile, true)
        } else {
          throw new RuntimeException(s"file $definitionPath already exists")
        }
      }
      val out = fs.create(prototxtFile)
      val byteArrayOut = new ByteArrayOutputStream()
      byteArrayOut.write(model.toString.getBytes)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } else {
      val prototxtFile = new java.io.File(definitionPath)
      if (prototxtFile.exists()) {
        if (overwrite) {
          prototxtFile.delete()
        } else {
          throw new RuntimeException(s"file $definitionPath already exists")
        }
      }
      val prototxtWriter = new OutputStreamWriter(new FileOutputStream(prototxtFile))
      prototxtWriter.write(model.toString)
      prototxtWriter.close
    }
  }

  private def cleantWeightAndBias(modelBuilder : BigDLModel.Builder): Unit = {
    modelBuilder.clearWeight
    modelBuilder.clearBias
    if (modelBuilder.getSubModulesCount > 0) {
      modelBuilder.clearSubModules
      modelBuilder.getSubModulesList.asScala.foreach(sub => {
        val subModelBuilder = BigDLModel.newBuilder(sub)
        cleantWeightAndBias(subModelBuilder)
        modelBuilder.addSubModules(subModelBuilder.build)
      })
    }
  }
}
