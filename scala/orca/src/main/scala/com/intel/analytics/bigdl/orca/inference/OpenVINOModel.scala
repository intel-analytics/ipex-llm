/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.inference

import java.io.{File, IOException}
import java.nio.file.{Files, Paths}
import java.util.{ArrayList, Arrays, UUID, List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.pipeline.api.net.{NetUtils, RegistryMap, SerializationHolder}
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import com.intel.analytics.zoo.pipeline.inference.OpenVINOModel.OpenVINOModelHolder
import org.apache.commons.io.FileUtils
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

class OpenVINOModel(var modelHolder: OpenVINOModelHolder,
                    var isInt8: Boolean,
                    var batchSize: Int = -1,
                    var deviceType: DeviceTypeEnumVal = DeviceType.CPU)
  extends AbstractModel with InferenceSupportive with Serializable {

  private var isRelease: Boolean = false

  @transient
  private lazy val supportive: OpenVinoInferenceSupportive = {
    OpenVINOModel.logger.info("Prepare OpenVinoInferenceSupportive")
    OpenVinoInferenceSupportive.forceLoad()
    new OpenVinoInferenceSupportive()
  }

  @transient
  private lazy val executableNetworkReference: Long = {
    OpenVINOModel.logger.info("Lazy loading OpenVINO model")
    var nativeRef = -1L
    try {
      val modelFile = File.createTempFile("OpenVINO", ".xml")
      Files.write(Paths.get(modelFile.toURI), modelHolder.modelBytes)
      val weightFile = File.createTempFile("OpenVINO", ".bin")
      Files.write(Paths.get(weightFile.toURI), modelHolder.weightBytes)

      nativeRef = if (isInt8) {
        OpenVINOModel.logger.info(s"Load int8 model")
        supportive.loadOpenVinoIRInt8(modelFile.getAbsolutePath,
          weightFile.getAbsolutePath,
          deviceType.value, batchSize)
      } else {
        OpenVINOModel.logger.info(s"Load fp32 model")
        supportive.loadOpenVinoIR(modelFile.getAbsolutePath,
          weightFile.getAbsolutePath,
          deviceType.value, batchSize)
      }
      FileUtils.deleteQuietly(modelFile)
      FileUtils.deleteQuietly(weightFile)
    }
    catch {
      case io: IOException =>
        OpenVINOModel.logger.error("error during loading OpenVINO model")
        throw io
    }
    nativeRef
  }

  override def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    val outputs = new ArrayList[JList[JTensor]]()
    inputs.asScala.map(input => {
      val tensor = input.get(0)
      val output = if (isInt8) {
        supportive.predictInt8(executableNetworkReference,
          tensor.getData, tensor.getShape)
      } else {
        supportive.predict(executableNetworkReference,
          tensor.getData, tensor.getShape)
      }
      outputs.add(Arrays.asList({
        output
      }))
    })
    outputs
  }

  override def predict(inputActivity: Activity): Activity = {
    val (inputList, batchSize) = inputActivity.isTable match {
      case true =>
        val inputTable = inputActivity.toTable
        val batchSize = inputTable.length()
        (transferBatchTableToJListOfJListOfJTensor(inputTable, batchSize), batchSize)
      case false =>
        val inputTensor = inputActivity.toTensor[Float]
        val batchSize = inputTensor.size(1)
        (transferBatchTensorToJListOfJListOfJTensor(inputTensor, batchSize), batchSize)
    }
    val outputs = predict(inputList)
    transferListOfActivityToActivityOfBatch(outputs, batchSize)
  }

  override def copy(num: Int): Array[AbstractModel] = Array(this)

  override def release(): Unit = {
    isReleased match {
      case true =>
      case false =>
        supportive.releaseOpenVINOIR(executableNetworkReference)
        isRelease = true
    }
  }

  override def isReleased(): Boolean = {
    isRelease
  }
}

object OpenVINOModel {

  private val modelBytesRegistry = new RegistryMap[(Array[Byte], Array[Byte])]()

  val logger = LoggerFactory.getLogger(getClass)

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class OpenVINOModelHolder(@transient var modelBytes: Array[Byte],
                            @transient var weightBytes: Array[Byte],
                            private var id: String)
    extends SerializationHolder {

    def this(modelBytes: Array[Byte], weightBytes: Array[Byte]) =
      this(modelBytes, weightBytes, UUID.randomUUID().toString)

    def getModelBytes(): Array[Byte] = {
      modelBytes
    }

    def getWeightBytes(): Array[Byte] = {
      weightBytes
    }

    override def writeInternal(out: CommonOutputStream): Unit = {
      val (graphDef, _) = modelBytesRegistry.getOrCreate(id) {
        (modelBytes, weightBytes)
      }
      logger.debug("Write OpenVINO model into stream")
      out.writeString(id)
      if (inDriver) {
        out.writeInt(graphDef._1.length)
        timing(s"writing " +
          s"${graphDef._1.length / 1024 / 1024}Mb openvino model to stream") {
          out.write(graphDef._1)
        }
        out.writeInt(graphDef._2.length)
        timing(s"writing " +
          s"${graphDef._2.length / 1024 / 1024}Mb openvino weight to stream") {
          out.write(graphDef._2)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (graphDef, _) = modelBytesRegistry.getOrCreate(id) {
        val modelLen = in.readInt()
        logger.debug("Read OpenVINO model from stream")
        assert(modelLen >= 0, "OpenVINO model length should be an non-negative integer")
        val localModelBytes = new Array[Byte](modelLen)
        timing("reading OpenVINO model from stream") {
          var numOfBytes = 0
          while (numOfBytes < modelLen) {
            val read = in.read(localModelBytes, numOfBytes, modelLen - numOfBytes)
            numOfBytes += read
          }
        }
        val weightLen = in.readInt()
        assert(weightLen >= 0, "OpenVINO weight length should be an non-negative integer")
        var localWeightBytes = new Array[Byte](weightLen)
        timing("reading OpenVINO weight from stream") {
          var numOfBytes = 0
          while (numOfBytes < weightLen) {
            val read = in.read(localWeightBytes, numOfBytes, weightLen - numOfBytes)
            numOfBytes += read
          }
        }
        (localModelBytes, localWeightBytes)
      }
      modelBytes = graphDef._1
      weightBytes = graphDef._2
      id = id
    }
  }

  def apply(modelHolder: OpenVINOModelHolder, isInt8: Boolean): OpenVINOModel = {
    new OpenVINOModel(modelHolder, isInt8)
  }

  def apply(modelHolder: OpenVINOModelHolder, isInt8: Boolean, batchSize: Int): OpenVINOModel = {
    new OpenVINOModel(modelHolder, isInt8, batchSize)
  }
}
