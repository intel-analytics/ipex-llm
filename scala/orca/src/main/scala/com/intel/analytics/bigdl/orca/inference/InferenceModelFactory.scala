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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.net.TFNet

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object InferenceModelFactory {

  def loadFloatInferenceModel(modelPath: String): FloatInferenceModel = {
    loadFloatInferenceModel(modelPath, null)
  }

  def loadFloatInferenceModel(modelPath: String, weightPath: String)
  : FloatInferenceModel = {
    val model = ModelLoader.loadFloatModel(modelPath, weightPath)
    model.evaluate()
    new FloatInferenceModel(model)
  }

  def loadFloatInferenceModelForCaffe(modelPath: String, weightPath: String)
  : FloatInferenceModel = {
    val model = ModelLoader.loadFloatModelForCaffe(modelPath, weightPath)
    model.evaluate()
    new FloatInferenceModel(model)
  }

  def loadFloatInferenceModelForTF(modelPath: String,
                                   intraOpParallelismThreads: Int = 1,
                                   interOpParallelismThreads: Int = 1,
                                   usePerSessionThreads: Boolean = true): FloatInferenceModel = {
    val sessionConfig = TFNet.SessionConfig(intraOpParallelismThreads,
      interOpParallelismThreads, usePerSessionThreads)
    val model = ModelLoader.loadFloatModelForTF(modelPath, sessionConfig)
    model.evaluate()
    new FloatInferenceModel(model)
  }

  def cloneSharedWeightsModelsIntoArray(originalModel: FloatInferenceModel,
                                                num: Int): Array[FloatInferenceModel] = {
    var modelList = ArrayBuffer[FloatInferenceModel]()
    val emptyModel = originalModel.model.cloneModule()
    clearWeightsBias(emptyModel)
    var i = 0
    while (i < num) {
      val clonedModel = emptyModel.cloneModule
      val newModel = makeUpModel(clonedModel, originalModel.model.getWeightsBias)
      modelList.append(newModel)
      i += 1
    }
    modelList.toArray
  }

  private def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])
                                      (implicit ev: TensorNumeric[T]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        tensors(i).set()
      }
      i += 1
    }
  }

  private def clearWeightsBias(model: Module[Float]): Unit = {
    clearTensor(model.parameters()._1)
    clearTensor(model.parameters()._2)
  }

  private def putWeightsBias(weightBias: Array[Tensor[Float]],
                             localModel: Module[Float]): Module[Float] = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(weightBias(i))
      }
      i += 1
    }
    localModel
  }

  private def makeUpModel(model: Module[Float], weightBias: Array[Tensor[Float]]):
  FloatInferenceModel = {
    val newModel = model.cloneModule()
    putWeightsBias(weightBias, newModel)
    newModel.evaluate()
    new FloatInferenceModel(newModel)
  }
}
