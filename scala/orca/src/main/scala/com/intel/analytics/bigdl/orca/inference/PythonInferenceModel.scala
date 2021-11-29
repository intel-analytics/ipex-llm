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

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import java.util.{List => JList}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonInferenceModel {

  def ofFloat(): PythonInferenceModel[Float] = new PythonInferenceModel[Float]()

  def ofDouble(): PythonInferenceModel[Double] = new PythonInferenceModel[Double]()
}

class PythonInferenceModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def createInferenceModel(supportedConcurrentNum: Int = 1): InferenceModel = {
    new InferenceModel(supportedConcurrentNum)
  }

  def inferenceModelLoadBigDL(
      model: InferenceModel,
      modelPath: String,
      weightPath: String): Unit = {
    model.doLoadBigDL(modelPath, weightPath)
  }

  @deprecated("this method is deprecated", "0.8.0")
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
      weightPath: String,
      batchSize: Int = 0): Unit = {
    model.doLoadOpenVINO(modelPath, weightPath, batchSize)
  }

  def inferenceModelLoadOpenVINONg(model: InferenceModel,
                                   modelPath: String,
                                   weightPath: String,
                                   batchSize: Int = 0): Unit = {
    model.doLoadOpenVINONg(modelPath, weightPath, batchSize)
  }

  def inferenceModelLoadTensorFlow(
      model: InferenceModel,
      modelPath: String,
      modelType: String,
      intraOpParallelismThreads: Int,
      interOpParallelismThreads: Int,
      usePerSessionThreads: Boolean): Unit = {
    model.doLoadTensorflow(modelPath, modelType, intraOpParallelismThreads,
      interOpParallelismThreads, usePerSessionThreads)
  }

  def inferenceModelLoadTensorFlow(
      model: InferenceModel,
      modelPath: String,
      modelType: String,
      inputs: JList[String],
      outputs: JList[String],
      intraOpParallelismThreads: Int,
      interOpParallelismThreads: Int,
      usePerSessionThreads: Boolean): Unit = {
    model.doLoadTensorflow(modelPath, modelType, Option(inputs).map(_.asScala.toArray).orNull,
      Option(outputs).map(_.asScala.toArray).orNull, intraOpParallelismThreads,
      interOpParallelismThreads, usePerSessionThreads)
  }

  def inferenceModelLoadPytorch(
      model: InferenceModel,
      modelBytes: Array[Byte]): Unit = {
    model.doLoadPyTorch(modelBytes)
  }

  def inferenceModelPredict(
      model: InferenceModel,
      inputs: JList[com.intel.analytics.bigdl.dllib.utils.python.api.JTensor],
      inputIsTable: Boolean): JList[Object] = {
    val inputActivity = jTensorsToActivity(inputs, inputIsTable)
    val outputActivity = model.doPredict(inputActivity)
    activityToList(outputActivity)
  }
}
