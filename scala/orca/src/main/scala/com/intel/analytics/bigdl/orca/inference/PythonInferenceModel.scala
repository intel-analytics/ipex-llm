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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import java.util.{List => JList}

import scala.reflect.ClassTag

object PythonInferenceModel {

  def ofFloat(): PythonInferenceModel[Float] = new PythonInferenceModel[Float]()

  def ofDouble(): PythonInferenceModel[Double] = new PythonInferenceModel[Double]()
}

class PythonInferenceModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def createInferenceModel(supportedConcurrentNum: Int = 1): InferenceModel = {
    new InferenceModel(supportedConcurrentNum)
  }

  def inferenceModelLoad(
      model: InferenceModel,
      modelPath: String,
      weightPath: String): Unit = {
    model.doLoad(modelPath, weightPath)
  }

  def inferenceModelLoadCaffe(
      model: InferenceModel,
      modelPath: String,
      weightPath: String): Unit = {
    model.doLoadCaffe(modelPath, weightPath)
  }

  def inferenceModelLoadOpenVINO(
      model: InferenceModel,
      modelPath: String,
      weightPath: String): Unit = {
    model.doLoadOpenVINO(modelPath, weightPath)
  }

  def inferenceModelOpenVINOLoadTF(
      model: InferenceModel,
      modelPath: String,
      modelType: String): Unit = {
    model.doLoadTF(modelPath, modelType)
  }

  def inferenceModelOpenVINOLoadTF(
      model: InferenceModel,
      modelPath: String,
      pipelineConfigFilePath: String,
      extensionsConfigFilePath: String): Unit = {
    model.doLoadTF(modelPath, pipelineConfigFilePath, extensionsConfigFilePath)
  }

  def inferenceModelOpenVINOLoadTF(model: InferenceModel,
                                   modelPath: String,
                                   modelType: String,
                                   pipelineConfigFilePath: String,
                                   extensionsConfigFilePath: String): Unit = {
    model.doLoadTF(modelPath, modelType, pipelineConfigFilePath, extensionsConfigFilePath)
  }

  def inferenceModelTensorFlowLoadTF(
      model: InferenceModel,
      modelPath: String,
      intraOpParallelismThreads: Int,
      interOpParallelismThreads: Int,
      usePerSessionThreads: Boolean): Unit = {
    model.doLoadTF(modelPath, intraOpParallelismThreads,
      interOpParallelismThreads, usePerSessionThreads)
  }

  def inferenceModelPredict(
      model: InferenceModel,
      inputs: JList[com.intel.analytics.bigdl.python.api.JTensor],
      inputIsTable: Boolean): JList[Object] = {
    val inputActivity = jTensorsToActivity(inputs, inputIsTable)
    val outputActivity = model.doPredict(inputActivity)
    activityToList(outputActivity)
  }
}
