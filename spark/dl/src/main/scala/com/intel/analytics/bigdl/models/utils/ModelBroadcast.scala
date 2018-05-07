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
import com.intel.analytics.bigdl.nn.{Container, Graph}
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
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
 * @param applyProtoBuffer it will use proto buffer serialization for broadcasting if set true
 */
class ModelBroadcast[T: ClassTag](applyProtoBuffer: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Serializable {

  private var broadcastModel: Broadcast[Module[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _


  /**
   * broadcast the model
   * first get and clear Const values from the model
   * then get and clear the weight and bias parameters from the model
   * finally broadcast Const values, the parameters and model(without parameters) separately
   * @param sc    SparkContext
   * @param model model to broadcast
   * @return this
   */
  def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    if (applyProtoBuffer) {
      broadcastModel = sc.broadcast(model)
    } else {
      // broadcast Consts
      if (model.isInstanceOf[Container[_, _, T]]) {
        val moduleConsts = getAndClearConsts(model.asInstanceOf[Container[_, _, T]])
        // TODO: broadcast Const, model structure and weight in the same broadcast.
        broadcastConsts = sc.broadcast(moduleConsts)
      }
      // broadcast weight and model
      val weightsBias = getAndClearWeightBias(model.parameters())
      broadcastModel = sc.broadcast(model.cloneModule())
      broadcastParameters = sc.broadcast(weightsBias)

      putWeightBias(weightsBias, model)
      initGradWeightBias(weightsBias, model)
    }
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
    if (applyProtoBuffer) {
      val localModel = broadcastModel.value.clone(false)
      if (initGradient) {
        initGradWeightBias(getWeightBias(localModel.parameters()), localModel)
      }
      localModel
    } else {
      val localModel = broadcastModel.value.cloneModule()
      // share weight
      putWeightBias(broadcastParameters.value, localModel)
      // share Consts
      if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
        putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
      }
      // init gradient
      if (initGradient) {
        initGradWeightBias(broadcastParameters.value, localModel)
      }
      localModel
    }
  }

  private def getWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]]))
  : Array[Tensor[T]] = {
    if (parameters._1.length != 0) {
      var i = 0
      val weightsBias = new Array[Tensor[T]](parameters._1.length)
      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
      val (isCompacted, storage) = if (!isQuantized) {
        val storage = Storage(parameters._1(0).storage.array())
        (parameters._1.map(_.nElement()).sum == storage.length(), storage)
      } else {
        (false, null)
      }

      // get weight and bias
      while (i < parameters._1.length) {
        if (parameters._1(i) != null) {
          val wb = parameters._1(i)
          wb.getTensorType match {
            case QuantizedType =>
              val quantTensor = wb.asInstanceOf[QuantizedTensor[T]]
              weightsBias(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            case _ =>
              weightsBias(i) = if (isCompacted) {
                Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
              } else {
                Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
              }
          }
          i += 1
        }
      }
      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }

}


object ModelBroadcast {
  def apply[@specialized(Float, Double) T: ClassTag](applyProtoBuffer: Boolean = false)
        (implicit ev: TensorNumeric[T]) : ModelBroadcast[T] = {
    new ModelBroadcast(applyProtoBuffer)
  }
}
