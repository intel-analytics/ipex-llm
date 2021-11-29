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

package com.intel.analytics.bigdl.orca.inference

import com.intel.analytics.zoo.core.openvino.OpenvinoNativeLoader
import com.intel.analytics.bigdl.orca.inference.DeviceType.DeviceTypeEnumVal
import com.intel.analytics.bigdl.orca.inference.OpenVINOModel.OpenVINOModelHolder
import org.slf4j.LoggerFactory

import java.io.{File, FileOutputStream, InputStream}
import java.nio.channels.Channels
import java.nio.file.{Files, Paths}
import java.util.{List => JList}
import scala.io.Source
import scala.language.postfixOps


class OpenVinoInferenceSupportive extends InferenceSupportive with Serializable {

  @native def loadOpenVinoIR(modelFilePath: String,
                             weightFilePath: String,
                             deviceTypeValue: Int,
                             batchSize: Int): Long

  @native def loadOpenVinoIRInt8(modelFilePath: String,
                                 weightFilePath: String,
                                 deviceTypeValue: Int,
                                 batchSize: Int): Long

  @native def predict(executableNetworkReference: Long,
                      data: Array[Float],
                      shape: Array[Int]): JList[JTensor]

  @native def predict(executableNetworkReference: Long,
                      data: JList[JTensor]): JList[JTensor]

  @native def predictInt8(executableNetworkReference: Long,
                      data: JList[JTensor]): JList[JTensor]

  @native def predictInt8(executableNetworkReference: Long,
                      data: Array[Float],
                      shape: Array[Int]): JList[JTensor]

  @native def predictInt8(executableNetworkReference: Long,
                      data: Array[Byte],
                      shape: Array[Int]): JList[JTensor]

  @native def releaseOpenVINOIR(executableNetworkReference: Long): Unit
}

object OpenVinoInferenceSupportive extends InferenceSupportive with Serializable {
  val logger = LoggerFactory.getLogger(getClass)

  timing("load native so for OpenVINO") {
    OpenvinoNativeLoader.load()
  }


  def loadOpenVinoIRFromTempDir(modelName: String, tempDir: String): OpenVINOModel = {
    val modelFilePath: String = s"$tempDir/$modelName.xml"
    val weightFilePath: String = s"$tempDir/$modelName.bin"
    val mappingFilePath: String = s"$tempDir/$modelName.mapping"

    timing(s"load OpenVINO IR from $modelFilePath, $weightFilePath, $mappingFilePath") {
      val modelFile = new File(modelFilePath)
      val weightFile = new File(weightFilePath)
      val mappingFile = new File(mappingFilePath)
      val model = (modelFile.exists(), weightFile.exists(), mappingFile.exists()) match {
        case (true, true, _) =>
          loadOpenVinoIR(modelFilePath, weightFilePath, DeviceType.CPU)
        case (_, _, _) => throw
          new InferenceRuntimeException("OpenVINO load from Temp dir error")
      }
      timing("delete temporary model files") {
        modelFile.delete()
        weightFile.delete()
        if(mappingFile.exists()) {mappingFile.delete()}
      }
      model
    }
  }


  def loadOpenVinoIR(modelFilePath: String,
                     weightFilePath: String,
                     deviceType: DeviceTypeEnumVal,
                     batchSize: Int = 0): OpenVINOModel = {
    timing("load OpenVINO IR") {
      val modelBytes = Files.readAllBytes(Paths.get(modelFilePath))
      val weightBytes = Files.readAllBytes(Paths.get(weightFilePath))
      val buffer = Source.fromBytes(modelBytes)
      // For OpenVINO model version 9 or previous, check statistics keyword
      // For OpenVINO model version 10 or later, check FakeQuantize keyword
      val isInt8 = buffer
        .getLines()
        .count(_ matches ".*statistics.*|.*FakeQuantize.*") > 0
      buffer.close()
      new OpenVINOModel(new OpenVINOModelHolder(modelBytes, weightBytes),
        isInt8, batchSize, deviceType)
    }
  }

  def loadOpenVinoNgIR(modelFilePath: String,
                     weightFilePath: String,
                     deviceType: DeviceTypeEnumVal,
                     batchSize: Int = 0): OpenVINOModelNg = {
    timing("load OpenVINO IR") {
      val modelBytes = Files.readAllBytes(Paths.get(modelFilePath))
      val weightBytes = Files.readAllBytes(Paths.get(weightFilePath))
      val buffer = Source.fromBytes(modelBytes)
      // For OpenVINO model version 9 or previous, check statistics keyword
      // For OpenVINO model version 10 or later, check FakeQuantize keyword
      val isInt8 = buffer
        .getLines()
        .count(_ matches ".*statistics.*|.*FakeQuantize.*") > 0
      buffer.close()
      new OpenVINOModelNg(new OpenVINOModelHolder(modelBytes, weightBytes),
        isInt8, batchSize, deviceType)
    }
  }

  def loadOpenVinoIR(modelBytes: Array[Byte],
                     weightBytes: Array[Byte],
                     deviceType: DeviceTypeEnumVal,
                     batchSize: Int): OpenVINOModel = {
    timing("load OpenVINO IR") {
      val buffer = Source.fromBytes(modelBytes)
      // For OpenVINO model version 9 or previous, check statistics keyword
      // For OpenVINO model version 10 or later, check FakeQuantize keyword
      val isInt8 = buffer
        .getLines()
        .count(_ matches ".*statistics.*|.*FakeQuantize.*") > 0
      buffer.close()
      new OpenVINOModel(new OpenVINOModelHolder(modelBytes, weightBytes),
        isInt8, batchSize, deviceType)
    }
  }

  def forceLoad(): Unit = {
    logger.info("Force native loader")
  }

  def load(path: String): Unit = {
    logger.info(s"start to load library: $path.")
    val inputStream = OpenVinoInferenceSupportive.getClass.getResourceAsStream(s"/${path}")
    val file = File.createTempFile("OpenVinoInferenceSupportiveLoader", path)
    val src = Channels.newChannel(inputStream)
    val dest = new FileOutputStream(file).getChannel
    dest.transferFrom(src, 0, Long.MaxValue)
    dest.close()
    src.close()
    val filePath = file.getAbsolutePath
    logger.info(s"loading library: $path from $filePath ...")
    try {
      System.load(filePath)
    } finally {
      file.delete()
    }
  }

  def writeFile(inputStream: InputStream, dirPath: String, filePath: String): File = {
    val file = new File(s"$dirPath/$filePath")
    val src = Channels.newChannel(inputStream)
    val dest = new FileOutputStream(file).getChannel
    dest.transferFrom(src, 0, Long.MaxValue)
    dest.close()
    src.close()
    logger.info(s"file output to ${file.getAbsoluteFile} ...")
    file
  }
}
