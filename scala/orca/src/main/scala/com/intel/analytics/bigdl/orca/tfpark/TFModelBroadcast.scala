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

package com.intel.analytics.bigdl.orca.tfpark

import java.io._
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.models.utils.{CachedModels, ModelBroadcast, ModelInfo}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.nn.mkldnn.{MklDnnLayer, TensorMMap}
import com.intel.analytics.bigdl.dllib.nn.tf.Const
import com.intel.analytics.bigdl.dllib.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.dllib.common.CheckedObjectInputStream
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.dllib.net.SerializationHolder
import com.intel.analytics.bigdl.orca.tfpark.Util._
import org.apache.commons.io.serialization.ValidatingObjectInputStream
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class TFModelBroadcast[T: ClassTag]()
                                   (implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {
  //  private type NativeType = (String, (Array[TensorMMap], Array[TensorMMap]))
  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _
  private var broadcastExtraParameters: Broadcast[Array[Tensor[T]]] = _
  //  private var broadcastParametersNative: Broadcast[Array[NativeType]] = _
  private var nodeNumber: Int = _
  private var coreNumber: Int = _

  private def setNodeAndCore(): Unit = {
    nodeNumber = Engine.nodeNumber()
    coreNumber = Engine.coreNumber()
  }

  /**
   * broadcast the model
   * first get and clear Const values from the model
   * then get and clear the weight and bias parameters from the model
   * finally broadcast Const values, the parameters and model(without parameters) separately
   *
   * @param sc    SparkContext
   * @param model model to broadcast
   * @return this
   */
  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    CachedModels.deleteAll(uuid) // delete the models on driver


    // broadcast Consts
    //    if (model.isInstanceOf[Container[_, _, T]]) {
    //      val moduleConsts = getAndClearConsts(model.asInstanceOf[Container[_, _, T]])
    //      // TODO: broadcast Const, model structure and weight in the same broadcast.
    //      broadcastConsts = sc.broadcast(moduleConsts)
    //    }
    // broadcast weight and model
    val weightsBias = getAndClearWeightBias(model.parameters())
    val extraParams = getAndClearExtraParameters(model.getExtraParameter())
    broadcastModel = sc.broadcast(ModelInfo[T](uuid, model))
    broadcastParameters = sc.broadcast(weightsBias)

    broadcastExtraParameters = sc.broadcast(extraParams)
    broadcastParameters = sc.broadcast(weightsBias)

    // For quantized model if we don't clone weightsBias, the original model will be released also
    // when we delete all models used in `ModelBroadcast`.
    putWeightBias(cloneParameters(weightsBias), model)
    initGradWeightBias(weightsBias, model)
    putExtraParams(extraParams, model)

    setNodeAndCore()
    this
  }

  /**
   * get the broadcast model
   * put the weight and bias back to the model
   *
   * @param initGradient If create a tensor for gradient when fetch the model. Please note that
   *                     the gradient is not needed in model inference
   * @return model
   */
  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
    Engine.setCoreNumber(coreNumber)
    //    Engine.setNodeAndCore(nodeNumber, coreNumber)
    CachedModels.deleteAll(this.uuid)

    val localModel = broadcastModel.value.model.cloneModule()
    val uuid = broadcastModel.value.uuid
    CachedModels.add(uuid, localModel)

    val parameters = if (shareWeight) {
      broadcastParameters.value
    } else {
      SerializationUtils.clone(broadcastParameters.value)
    }

    // share weight
    putWeightBias(parameters, localModel)

    //    // share Consts
    //    if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
    //      putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
    //    }
    if (initGradient) {
      initGradWeightBias(broadcastParameters.value, localModel)
    }

    putExtraParams(broadcastExtraParameters.value, localModel)

    localModel
  }

  override def broadcast(sc: SparkContext, model: Module[T],
                         dummyInput: Activity): this.type = {
    this.broadcast(sc, model)
    this
  }

  override def value(initGradient: Boolean, shareWeight: Boolean,
                     dummyInput: Activity): Module[T] = {
    val model = value(initGradient, shareWeight)
    model
  }
}

private[bigdl] class ModelInfo[T: ClassTag](var uuid: String, @transient var model: Module[T])(
  implicit ev: TensorNumeric[T]) extends SerializationHolder {

  override def writeInternal(out: CommonOutputStream): Unit = {
    out.writeString(uuid)
    val stream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    val cloned = model.cloneModule()
    oos.writeObject(cloned)
    oos.close()
    val w = stream.toByteArray
    val len = w.length
    out.writeInt(len)
    out.write(w)
    CachedModels.add(uuid, cloned)
  }

  override def readInternal(in: CommonInputStream): Unit = {
    uuid = in.readString()
    val len = in.readInt()
    Log4Error.invalidOperationError(len != 0, "model length should not be zero," +
      "please set logging level to debug for more information")
    Log4Error.invalidOperationError(len >= 0, "model length should be an non-negative integer")
    val w = new Array[Byte](len)
    var numOfBytes = 0
    while (numOfBytes < len) {
      val read = in.read(w, numOfBytes, len - numOfBytes)
      numOfBytes += read
    }
    // val ois = new CheckedObjectInputStream(classOf[Module[T]], new ByteArrayInputStream(w))
    val ois = new ValidatingObjectInputStream(new ByteArrayInputStream(w))
    ois.accept(classOf[Module[T]])
    try {
      model = ois.readObject().asInstanceOf[Module[T]]
      CachedModels.add(uuid, model)
    } finally {
      ois.close()
    }
  }
}

private[bigdl] object ModelInfo {
  def apply[T: ClassTag](uuid: String, model: Module[T])(
    implicit ev: TensorNumeric[T]): ModelInfo[T] = new ModelInfo[T](uuid, model)
}

private[bigdl] object CachedModels {

  import java.util.concurrent.ConcurrentHashMap

  import scala.collection._
  import scala.collection.convert.decorateAsScala._
  import scala.language.existentials

  type Modles = ArrayBuffer[Module[_]]

  private val cachedModels =
    new ConcurrentHashMap[String, Modles]().asScala

  def add[T: ClassTag](uuid: String, model: Module[T])(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val models = cachedModels.get(uuid) match {
        case Some(values) => values += model.asInstanceOf[Module[_]]
        case _ => ArrayBuffer(model.asInstanceOf[Module[_]])
      }
      cachedModels.put(uuid, models.asInstanceOf[Modles])
    }

  def deleteAll[T: ClassTag](currentKey: String)(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (key <- keys) {
        if (key != currentKey) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }

  def deleteKey[T: ClassTag](key: String)(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (k <- keys) {
        if (k == key) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }
}

object Util {

  private[bigdl] def getAndClearWeightBias[T: ClassTag]
  (parameters: (Array[Tensor[T]], Array[Tensor[T]]))(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    clearTensor(parameters._2)
    getAndClearParameters(parameters._1)
  }

  private[bigdl] def getAndClearExtraParameters[T: ClassTag]
  (parameters: Array[Tensor[T]])(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    getAndClearParameters(parameters)
  }

  private[bigdl] def getAndClearParameters[T: ClassTag]
  (parameters: Array[Tensor[T]])(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    if (parameters != null) {
      if (parameters.length != 0) {
        var i = 0
        val retParams = new Array[Tensor[T]](parameters.length)
        //      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
        val (isCompacted, storage) = {
          val storage = Storage(parameters(0).storage.array())
          (parameters.map(_.nElement()).sum == storage.length(), storage)
        }

        // get parameters
        while (i < parameters.length) {
          if (parameters(i) != null) {
            val wb = parameters(i)
            retParams(i) = if (isCompacted) {
              Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
            } else {
              Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
            }
            i += 1
          }
        }
        // clear parameters
        clearTensor(parameters)

        retParams
      } else {
        // just return an empty array when parameters is empty.
        Array()
      }
    } else {
      null
    }
  }


  private def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])
                                      (implicit ev: TensorNumeric[T]): Unit = {
    if (tensors != null) {
      var i = 0
      while (i < tensors.length) {
        if (tensors(i) != null) {
          tensors(i).set()
        }
        i += 1
      }
    }
  }

  private[bigdl] def putWeightBias[T: ClassTag](broadcastWeightBias: Array[Tensor[T]],
                                              localModel: Module[T])(
                                               implicit ev: TensorNumeric[T]): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        clearAndSet(localWeightBias(i), broadcastWeightBias(i))
      }
      i += 1
    }

    def clearAndSet(old: Tensor[T], other: Tensor[T]): Unit = {
      old.set(other)
    }
  }

  private[bigdl] def putExtraParams[T: ClassTag](broadcastExtraParams: Array[Tensor[T]],
                                               localModel: Module[T])(
                                                implicit ev: TensorNumeric[T]): Unit = {
    val localExtraParams = localModel.getExtraParameter()
    if (localExtraParams != null) {
      var i = 0
      while (i < localExtraParams.length) {
        if (localExtraParams(i) != null) {
          localExtraParams(i).set(broadcastExtraParams(i))

        }
        i += 1
      }
    }

  }

  private[bigdl] def initGradWeightBias[T: ClassTag](broadcastWeightBias: Array[Tensor[T]],
                                                   localModel: Module[T])(
                                                    implicit ev: TensorNumeric[T]): Unit = {
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

  private[bigdl] def cloneParameters[T: ClassTag]
  (parameters: Array[Tensor[T]])(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    if (parameters != null) {
      if (parameters.length != 0) {
        var i = 0
        val retParams = new Array[Tensor[T]](parameters.length)

        val (isCompacted, storage) = {
          val storage = Storage(parameters(0).storage.array())
          (parameters.map(_.nElement()).sum == storage.length(), storage)
        }

        val resultStorage = if (isCompacted) {
          val resultStorage = Storage[T](storage.length())
          System.arraycopy(storage.array(), parameters(0).storageOffset() - 1,
            resultStorage.array(), 0, storage.length())
          resultStorage
        } else {
          null
        }

        // clone parameters
        while (i < parameters.length) {
          if (parameters(i) != null) {
            val wb = parameters(i)
            retParams(i) = if (isCompacted) {
              Tensor[T](resultStorage, wb.storageOffset(), wb.size(), wb.stride())
            } else {
              wb.clone()
            }
            i += 1
          }
        }

        retParams
      } else {
        // just return an empty array when parameters is empty.
        Array()
      }
    } else {
      null
    }
  }

}
