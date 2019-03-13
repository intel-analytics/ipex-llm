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

import java.io._
import java.util.{List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class FloatModel(var model: AbstractModule[Activity, Activity, Float])
  extends AbstractModel with InferenceSupportive with Serializable {

  override def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    val batchSize = inputs.size()
    require(batchSize > 0, "inputs size should > 0")

    val inputActivity = transferListOfActivityToActivityOfBatch(inputs, batchSize)
    val result: Activity = predict(inputActivity)

    val outputs = result.isTensor match {
      case true =>
        val outputTensor = result.toTensor[Float]
        transferBatchTensorToJListOfJListOfJTensor(outputTensor, batchSize)
      case false =>
        val outputTable: Table = result.toTable
        transferBatchTableToJListOfJListOfJTensor(outputTable, batchSize)
    }
    outputs
  }

  override def predict(inputActivity: Activity): Activity = {
    model.forward(inputActivity)
  }

  override def copy(num: Int): Array[AbstractModel] = {
    cloneSharedWeightsModelsIntoArray(this, num)
  }

  override def release(): Unit = {
    isReleased match {
      case true =>
      case false => model.release(); model = null
    }
  }

  override def isReleased(): Boolean = {
    model == null
  }

  override def toString: String = s"FloatInferenceModel($model)"

  def cloneSharedWeightsModelsIntoArray(originalModel: FloatModel, num: Int):
  Array[AbstractModel] = {
    var modelList = ArrayBuffer[FloatModel]()
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

  private def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]):
  Unit = {
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

  private def putWeightsBias(weightBias: Array[Tensor[Float]], localModel: Module[Float]):
  Module[Float] = {
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
  FloatModel = {
    val newModel = model.cloneModule()
    putWeightsBias(weightBias, newModel)
    newModel.evaluate()
    new FloatModel(newModel)
  }
}
