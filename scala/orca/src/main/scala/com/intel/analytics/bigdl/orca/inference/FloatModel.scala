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

class FloatModel(var model: AbstractModule[Activity, Activity, Float],
                 var metaModel: AbstractModule[Activity, Activity, Float],
                 var isOriginal: Boolean)
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
    doCopy(metaModel, model.getWeightsBias(), num)
  }

  override def release(): Unit = {
    isReleased match {
      case true =>
      case false =>
        model.release()
        model = null
        metaModel = null
    }
  }

  override def isReleased(): Boolean = {
    model == null
  }

  override def toString: String = s"FloatInferenceModel($model)"

  def doCopy(metaModel: AbstractModule[Activity, Activity, Float],
             weightBias: Array[Tensor[Float]],
             num: Int):
  Array[AbstractModel] = {
    require(metaModel != null, "metaModel can NOT be null")
    List.range(0, num).map(_ => {
      val clonedModel = metaModel.cloneModule()
      val clonedModelWithWeightsBias = makeUpModel(clonedModel, weightBias)
      new FloatModel(clonedModelWithWeightsBias, metaModel, false)
    }).toArray
  }
}
