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
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

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


  private def getAndClearWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]]))
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
      // clear parameters
      clearTensor(parameters._1)
      // because in quantized mode, the weight number may be different with gradWeight number
      clearTensor(parameters._2)

      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }

  private def clearTensor(tensors: Array[Tensor[T]]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        tensors(i).set()
      }
      i += 1
    }
  }

  private def putWeightBias(
        broadcastWeightBias: Array[Tensor[T]],
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

  private def initGradWeightBias(
        broadcastWeightBias: Array[Tensor[T]],
        localModel: Module[T]): Unit = {
    val (localWeightBias, localGradWeightBias) = localModel.parameters()
    // init gradient with a compacted storage
    val storage = Storage[T](localGradWeightBias.map(_.nElement()).sum)
    val isQuantized = broadcastWeightBias.exists(_.getTensorType == QuantizedType)
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        val wb = broadcastWeightBias(i)
        wb.getTensorType match {
          case QuantizedType =>
            localGradWeightBias(i).set(Tensor(1))
          case _ =>
            localGradWeightBias(i).set(storage, wb.storageOffset(), wb.size(), wb.stride())
        }
      }
      i += 1
    }
  }
}


object ModelBroadcast {
  def apply[@specialized(Float, Double) T: ClassTag]()
        (implicit ev: TensorNumeric[T]) : ModelBroadcast[T] = {
    new ModelBroadcast()
  }
}
