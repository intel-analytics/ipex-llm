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

import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal

object InferenceModelFactory extends InferenceSupportive {

  def loadFloatModel(modelPath: String): FloatModel = {
    loadFloatModel(modelPath, null)
  }

  def loadFloatModel(modelPath: String, weightPath: String)
  : FloatModel = {
    val model = ModelLoader.loadFloatModel(modelPath, weightPath)
    model.evaluate()
    val metaModel = makeMetaModel(model)
    new FloatModel(model, metaModel, true)
  }

  def loadFloatModelForCaffe(modelPath: String, weightPath: String): FloatModel = {
    val model = ModelLoader.loadFloatModelForCaffe(modelPath, weightPath)
    model.evaluate()
    val metaModel = makeMetaModel(model)
    new FloatModel(model, metaModel, true)
  }

  def loadFloatModelForTF(modelPath: String,
                          intraOpParallelismThreads: Int = 1,
                          interOpParallelismThreads: Int = 1,
                          usePerSessionThreads: Boolean = true): FloatModel = {
    val sessionConfig = TFNet.SessionConfig(intraOpParallelismThreads,
      interOpParallelismThreads, usePerSessionThreads)
    val model = ModelLoader.loadFloatModelForTF(modelPath, sessionConfig)
    model.evaluate()
    val metaModel = makeMetaModel(model)
    new FloatModel(model, metaModel, true)
  }

  def loadOpenVINOModelForTF(modelPath: String,
                             modelType: String,
                             pipelineConfigPath: String,
                             extensionsConfigPath: String): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadTensorflowModel(
      modelPath, modelType, pipelineConfigPath, extensionsConfigPath)
  }

  def loadOpenVINOModelForTF(modelPath: String,
                             imageClassificationModelType: String,
                             checkpointPath: String,
                             inputShape: Array[Int],
                             ifReverseInputChannels: Boolean,
                             meanValues: Array[Float],
                             scale: Float): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadTensorflowModel(modelPath, imageClassificationModelType,
      checkpointPath, inputShape, ifReverseInputChannels, meanValues, scale)
  }

  def loadOpenVINOModelForTF(modelBytes: Array[Byte],
                             imageClassificationModelType: String,
                             checkpointBytes: Array[Byte],
                             inputShape: Array[Int],
                             ifReverseInputChannels: Boolean,
                             meanValues: Array[Float],
                             scale: Float): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadTensorflowModel(modelBytes, imageClassificationModelType,
      checkpointBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }

  def loadOpenVINOModelForTF(savedModelDir: String,
                             inputShape: Array[Int],
                             ifReverseInputChannels: Boolean,
                             meanValues: Array[Float],
                             scale: Float,
                             input: String): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadTensorflowModel(savedModelDir,
      inputShape, ifReverseInputChannels, meanValues, scale, input)
  }

  def loadOpenVINOModelForTF(savedModelBytes: Array[Byte],
                             inputShape: Array[Int],
                             ifReverseInputChannels: Boolean,
                             meanValues: Array[Float],
                             scale: Float,
                             input: String): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadTensorflowModel(savedModelBytes,
      inputShape, ifReverseInputChannels, meanValues, scale, input)
  }

  def loadCalibratedOpenVINOModelForTF(modelPath: String,
                                       modelType: String,
                                       checkpointPath: String,
                                       inputShape: Array[Int],
                                       ifReverseInputChannels: Boolean,
                                       meanValues: Array[Float],
                                       scale: Float,
                                       networkType: String,
                                       validationFilePath: String,
                                       subset: Int,
                                       opencvLibPath: String): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadTensorflowModelAsCalibrated(
      modelPath, modelType, checkpointPath,
      inputShape, ifReverseInputChannels, meanValues, scale,
      networkType, validationFilePath, subset, opencvLibPath)
  }

  def loadOpenVINOModelForIR(modelFilePath: String,
                             weightFilePath: String,
                             deviceType: DeviceTypeEnumVal,
                             batchSize: Int = 0): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadOpenVinoIR(modelFilePath, weightFilePath,
      deviceType, batchSize)
  }

  def loadOpenVINOModelForIR(modelBytes: Array[Byte],
                             weightBytes: Array[Byte],
                             deviceType: DeviceTypeEnumVal,
                             batchSize: Int): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadOpenVinoIR(modelBytes, weightBytes,
      deviceType, batchSize)
  }
}
