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

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import java.util.UUID

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.quantized.StorageManager
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Util._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer
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

  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _

  val uuid: String = UUID.randomUUID().toString

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
    CachedModels.deleteAll(uuid) // delete the models on driver

    if (applyProtoBuffer) {
      broadcastModel = sc.broadcast(ModelInfo(uuid, model))
    } else {
      // We should clone a new model which will maintain the origin model.
      // Otherwise, the origin model's resources will be cleaned.
      val newModel = model.cloneModule()
      CachedModels.add(uuid, newModel)

      // broadcast Consts
      if (newModel.isInstanceOf[Container[_, _, T]]) {
        val moduleConsts = getAndClearConsts(newModel.asInstanceOf[Container[_, _, T]])
        // TODO: broadcast Const, model structure and weight in the same broadcast.
        broadcastConsts = sc.broadcast(moduleConsts)
      }

      // broadcast weight and model
      val weightsBias = getAndClearWeightBias(newModel.parameters())

      // We broadcast weight and model separately because of the memory limit of serialization.
      // And we should clone the model structure (without weight) first because of lazy evaluation
      // of broadcast. As you see, we have to put weights back to the model after broadcast call.
      // As a quantized model, it will create relevant memory after clone because of
      // `QuantizedTensor`. So we should release it first.
      val cloned = newModel.cloneModule()
      cloned.release()
      CachedModels.add(uuid, cloned)

      broadcastModel = sc.broadcast(ModelInfo[T](uuid, cloned))
      broadcastParameters = sc.broadcast(weightsBias)

      putWeightBias(weightsBias, newModel)
      initGradWeightBias(weightsBias, newModel)
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
    CachedModels.deleteAll(uuid)
    if (applyProtoBuffer) {
      val localModel = broadcastModel.value.model.clone(false)
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)

      if (initGradient) {
        initGradWeightBias(getWeightBias(localModel.parameters()), localModel)
      }
      localModel
    } else {
      val localModel = broadcastModel.value.model.cloneModule()
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)

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

class ModelInfo[T: ClassTag](val uuid: String, @transient var model: Module[T])(
  implicit ev: TensorNumeric[T]) extends Serializable {
  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    val cloned = model.cloneModule()
    out.writeObject(cloned)
    CachedModels.add(uuid, cloned)
  }
}

object ModelInfo {
  def apply[T: ClassTag](uuid: String, model: Module[T])(
    implicit ev: TensorNumeric[T]): ModelInfo[T] = new ModelInfo[T](uuid, model)
}

object CachedModels {
  import java.util.concurrent.ConcurrentHashMap

  import scala.collection._
  import scala.collection.convert.decorateAsScala._

  val lock = new AnyRef

  type FloatValues = ArrayBuffer[Module[Float]]
  type DoubleValues = ArrayBuffer[Module[Double]]

  private val cachedFloatModels: concurrent.Map[String, FloatValues] =
    new ConcurrentHashMap[String, FloatValues]().asScala
  private val cachedDoubleModels: concurrent.Map[String, DoubleValues] =
    new ConcurrentHashMap[String, DoubleValues]().asScala

  def add[T: ClassTag](uuid: String, model: Module[T])( implicit ev: TensorNumeric[T]): Unit =
    lock.synchronized {
      ev.getType() match {
        case FloatType =>
          val models = cachedFloatModels.get(uuid) match {
            case Some(values) => values += model.asInstanceOf[Module[Float]]
            case _ => ArrayBuffer(model.asInstanceOf[Module[Float]])
          }
          cachedFloatModels.put(uuid, models)
        case DoubleType =>
          val models = cachedDoubleModels.get(uuid) match {
            case Some(values) => values += model.asInstanceOf[Module[Double]]
            case _ => ArrayBuffer(model.asInstanceOf[Module[Double]])
          }
          cachedDoubleModels.put(uuid, models)
        case _ => throw new UnsupportedOperationException(s"unsupported type")
      }
    }

  def deleteAll[T: ClassTag](currentKey: String)(implicit ev: TensorNumeric[T]): Unit =
    lock.synchronized {
      ev.getType() match {
        case FloatType =>
          val keys = cachedFloatModels.keys
          for (key <- keys) {
            if (key != currentKey) {
              val models = cachedFloatModels(key)
              println(s"delete key = $key ${models.length}")
              for (model <- models) { model.release()
                println(StorageManager.get().count(!_._2.isFreed))
              }
              cachedFloatModels.remove(key)
              println(StorageManager.get().count(!_._2.isFreed))
            }
          }
        case DoubleType =>
          val keys = cachedDoubleModels.keys
          for (key <- keys) {
            if (key != currentKey) {
              val models = cachedDoubleModels(key)
              for (model <- models) { model.release() }
              cachedDoubleModels.remove(key)
            }
          }
        case _ => throw new UnsupportedOperationException(s"unsupported type")
      }
    }
}
