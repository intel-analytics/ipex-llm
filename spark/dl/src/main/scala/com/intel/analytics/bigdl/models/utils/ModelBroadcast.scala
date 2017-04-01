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

package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

/**
 * ModelBroadcast is used to broadcast model when doing model inference.
 * Note: do not use it in model training since the broadcast models share weights and biases
 * It shortens the broadcast time, which is especially useful when the model size is large
 * @tparam T data type
 */
class ModelBroadcast[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {

  private var broadcastModel: Broadcast[Module[T]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _

  /**
   * broadcast the model
   * first get and clear the weight and bias parameters from the model
   * then broadcast the parameters and model(without parameters) separately
   * @param sc    SparkContext
   * @param model model to broadcast
   * @return this
   */
  def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    val bcModel = model.cloneModule()
    val weightsBias = getAndClearWeightBias(bcModel.parameters())
    broadcastModel = sc.broadcast(bcModel)
    broadcastParameters = sc.broadcast(weightsBias)
    this
  }

  /**
   * get the broadcast model
   * put the weight and bias back to the model
   * @return model
   */
  def value(): Module[T] = {
    val localModel = broadcastModel.value.cloneModule()
    putWeightBias(broadcastParameters.value, localModel)
    localModel
  }


  private def getAndClearWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]]))
  : Array[Tensor[T]] = {
    var i = 0
    val weightsBias = new Array[Tensor[T]](parameters._1.length)
    while (i < parameters._1.length) {
      if (parameters._1(i) != null) {
        val wb = parameters._1(i)
        weightsBias(i) = Tensor[T](Storage(wb.storage().array()),
          wb.storageOffset(), wb.size(), wb.stride())
      }
      i += 1
    }
    i = 0
    while (i < parameters._1.length) {
      if (parameters._1(i) != null) {
        parameters._1(i).set()
      }
      if (parameters._2(i) != null) {
        parameters._2(i).set()
      }
      i += 1
    }
    weightsBias
  }

  private def putWeightBias(broadcastWeightBias: Array[Tensor[T]],
    localModel: Module[T]): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(broadcastWeightBias(i))
      }
      i += 1
    }
  }
}


object ModelBroadcast {
  def apply[@specialized(Float, Double) T: ClassTag]()(implicit ev: TensorNumeric[T])
  : ModelBroadcast[T] = new ModelBroadcast
}
