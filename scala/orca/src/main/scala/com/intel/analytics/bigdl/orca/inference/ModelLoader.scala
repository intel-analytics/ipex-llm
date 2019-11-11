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

import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.pipeline.api.keras.layers.WordEmbedding
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.net.{GraphNet, TFNet}
import org.slf4j.LoggerFactory

object ModelLoader extends InferenceSupportive {
  val logger = LoggerFactory.getLogger(getClass)

  Model
  Sequential
  GraphNet
  WordEmbedding

  timing("bigdl init engine") {
    System.setProperty("bigdl.localMode", System.getProperty("bigdl.localMode", "true"))
    System.setProperty("bigdl.coreNumber", System.getProperty("bigdl.coreNumber", "1"))
    Engine.init
  }

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
}

