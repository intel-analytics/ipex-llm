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

import java.io.{ByteArrayInputStream, File, FileOutputStream}
import java.nio.channels.Channels

import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.pipeline.api.keras.layers.WordEmbedding
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.net.{GraphNet, TFNet, TorchNet}
import org.slf4j.LoggerFactory

import scala.language.postfixOps
import sys.process._

object ModelLoader extends InferenceSupportive {
  val logger = LoggerFactory.getLogger(getClass)

  Model
  Sequential
  GraphNet
  WordEmbedding

  def loadFloatModel(modelPath: String, weightPath: String)
    : AbstractModule[Activity, Activity, Float] = {
    timing(s"load model") {
      logger.info(s"load model from $modelPath and $weightPath")
      val model = ModuleLoader.loadFromFile[Float](modelPath, weightPath)
      logger.info(s"loaded model as $model")
      model
    }
  }

  def loadFloatModelForCaffe(modelPath: String, weightPath: String)
    : AbstractModule[Activity, Activity, Float] = {
    timing(s"load model") {
      logger.info(s"load model from $modelPath and $weightPath")
      val model = CaffeLoader.loadCaffe[Float](modelPath, weightPath)._1.asInstanceOf[Graph[Float]]
      logger.info(s"loaded model as $model")
      model
    }
  }

  def loadFloatModelForTF(modelPath: String,
                          config: TFNet.SessionConfig = TFNet.defaultSessionConfig)
  : AbstractModule[Activity, Activity, Float] = {
    timing("load model") {
      logger.info(s"load model from $modelPath")
      val model = TFNet(modelPath, config)
      logger.info(s"loaded model as $model")
      model
    }
  }

  def loadFloatModelForTFSavedModel(modelPath: String,
                                    inputs: Array[String],
                                    outputs: Array[String],
                                    config: TFNet.SessionConfig = TFNet.defaultSessionConfig)
  : AbstractModule[Activity, Activity, Float] = {
    timing("load model") {
      logger.info(s"load model from $modelPath")
      val model = TFNet.fromSavedModel(modelPath, inputs, outputs)
      logger.info(s"loaded model as $model")
      model
    }
  }

  def loadFloatModelForTFSavedModelBytes(savedModelBytes: Array[Byte],
                                         inputs: Array[String],
                                         outputs: Array[String],
                                         config: TFNet.SessionConfig = TFNet.defaultSessionConfig)
  : AbstractModule[Activity, Activity, Float] = {
    timing("load model") {
      logger.info(s"load model from $savedModelBytes")
      val tmpDir = Utils.createTmpDir("ZOOTFNet").toFile()
      val outputPath: String = tmpDir.getCanonicalPath

      val tarFilePath = (savedModelBytes == null) match {
        case true => null
        case false => val tarFileName = "saved-model.tar"
          val tarFile = new File(s"$tmpDir/$tarFileName")
          val tarFileInputStream = new ByteArrayInputStream(savedModelBytes)
          val tarFileSrc = Channels.newChannel(tarFileInputStream)
          val tarFileDest = new FileOutputStream(tarFile).getChannel
          tarFileDest.transferFrom(tarFileSrc, 0, Long.MaxValue)
          tarFileDest.close()
          tarFileSrc.close()
          tarFile.getAbsolutePath
      }
      s"mkdir -p $tmpDir/saved-model" !;
      s"tar xf $tarFilePath -C $tmpDir/saved-model" !;
      s"ls $tmpDir/saved-model" !;
      val savedModelDir = new File(s"$tmpDir/saved-model").listFiles()(0).getAbsolutePath

      val model = TFNet.fromSavedModel(savedModelDir, inputs, outputs)
      logger.info(s"loaded model as $model")
      s"rm -rf $tmpDir" !;
      model
    }
  }

  def loadFloatModelForPyTorch(modelPath: String)
  : AbstractModule[Activity, Activity, Float] = {
    timing("load model") {
      logger.info(s"load model from $modelPath")
      val model = TorchNet(modelPath)
      logger.info(s"loaded model as $model")
      model
    }
  }

  def loadFloatModelForPyTorch(modelBytes: Array[Byte])
  : AbstractModule[Activity, Activity, Float] = {
    timing("load model") {
      logger.info(s"load model from $modelBytes")
      val model = TorchNet(modelBytes)
      logger.info(s"loaded model as $model")
      model
    }
  }
}

