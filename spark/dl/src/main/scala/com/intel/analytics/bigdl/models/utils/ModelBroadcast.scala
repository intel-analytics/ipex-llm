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
import com.intel.analytics.bigdl.tensor.{Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import com.intel.analytics.bigdl.utils.Util._

import scala.reflect.ClassTag

/**
 * ModelBroadcast is used to broadcast model.
 *
 * Note: If you want to use this to broadcast training model, please use value(true) to get
 * the model. And before broadcasting please make sure the model's parameter is compacted.
 *
 * @tparam T data type
 */
class ModelBroadcast[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Serializable {

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
    val weightsBias = getAndClearWeightBias(model.parameters())
    broadcastModel = sc.broadcast(model.cloneModule())
    broadcastParameters = sc.broadcast(weightsBias)
    putWeightBias(weightsBias, model)
    initGradWeightBias(weightsBias, model)
    this
  }

  /**
   * get the broadcast model
   * put the weight and bias back to the model
   *
   * @param initGradient if init gradParameter.
   * @return model
   */
  def value(initGradient: Boolean = false): Module[T] = {
    val localModel = broadcastModel.value.cloneModule()
    putWeightBias(broadcastParameters.value, localModel)
    if (initGradient) {
      initGradWeightBias(broadcastParameters.value, localModel)
    }
    localModel
  }
}


object ModelBroadcast {
  def apply[@specialized(Float, Double) T: ClassTag]()
        (implicit ev: TensorNumeric[T]) : ModelBroadcast[T] = {
    new ModelBroadcast()
  }
}
