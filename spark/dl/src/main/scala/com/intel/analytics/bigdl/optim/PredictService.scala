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

package com.intel.analytics.bigdl.optim

import java.util.concurrent.{Executors, LinkedBlockingQueue}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue.ArrayValue
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLTensor, DataType, TensorStorage}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericChar, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericString}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializer, ProtoStorageType, SerializeContext}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.concurrent.{ExecutionContext, ExecutionContextExecutor, Future}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.Type

/**
 * <h6>Predict Service Interface for Concurrent Calls</h6>
 * In this service, several `model instances` sharing weights/bias will be built.
 * <br>
 * A `BlockingQueue` will be built to maintain available `model instances`.
 * <br>
 * When predict method called, service will try to take an instance from `BlockingQueue`,
 * which means if all instances are on serving, the predicting request will be blocked until
 * some instances are released.
 * <br>
 * The release method will be executed asynchronously when predictions finished.
 *
 * @param model BigDL model used to do predictions
 * @param nInstances number of model instances
 * @param nBackendThreads number of threads to release instances after predictions.
 */
class PredictService[T: ClassTag] private[optim](
  model: Module[T],
  nInstances: Int = 10,
  nBackendThreads: Int = 2
)(implicit ev: TensorNumeric[T]) {

  require(nInstances > 1, "At least two instances are needed!")

  require(nBackendThreads >= 1, "At least one backend threads are needed!")

  // ThreadPool for scala.concurrent.Future,
  // which is build for keeping threads available for unlocking to avoid dead locks.
  private implicit val ec: ExecutionContextExecutor = {
    ExecutionContext.fromExecutor(Executors.newFixedThreadPool(nBackendThreads))
  }

  protected val instQueue: LinkedBlockingQueue[Module[T]] = {
    val shallowCopies = (1 to nInstances)
      .map(_ => model.clone(false).evaluate()).asJava

    new LinkedBlockingQueue[Module[T]](shallowCopies)
  }

  def predict(request: Activity): Activity = {
    val module = fetchInstance()
    val output = module.forward(request)
    releaseInstance(module)
    output
  }

  def predict(request: Array[Byte]): Array[Byte] = {
    val activity = PredictService.deSerializeActivity(request)
    val output = predict(activity)
    val bytesOut = PredictService.serializeActivity(output)
    bytesOut
  }

  // Take an instance from blocking queue,
  // it will cause a thread blocking when no instance is available.
  protected def fetchInstance(): Module[T] = {
    instQueue.take()
  }

  // Asynchronously release module instance back to blocking queue.
  protected def releaseInstance(inst: Module[T]): Future[Boolean] = {
    Future {
      val isSuccess = instQueue.offer(inst)
      println(s"Release instance ${if (isSuccess) "successfully" else "failed"}!" +
        s" ${instQueue.size()} instances are available now!")
      isSuccess
    }
  }

}


object PredictService {

  def apply[T: ClassTag](
    model: Module[T],
    nInstances: Int = 10,
    nBackendThreads: Int = 2
  )(implicit ev: TensorNumeric[T]): PredictService[T] = {
    new PredictService[T](model, nInstances, nBackendThreads)
  }

  /**
   * <h6>Serialize activities to Array[Byte] according to `Bigdl.proto`.</h6>
   * For now, `Tensor` and `Table[Int, Tensor]` are supported.
   *
   * @param activity activity to be serialized
   */
  def serializeActivity(activity: Activity): Array[Byte] = {
    val attrBuilder = AttrValue.newBuilder()
    activity match {
      case table: Table =>
        val tensors = table.getState().toArray.sortBy(_._1.toString.toInt)
          .map(_._2.asInstanceOf[Tensor[_]])

        val arrayValue = ArrayValue.newBuilder
        arrayValue.setDatatype(DataType.TENSOR)
        arrayValue.setSize(tensors.length)
        tensors.foreach { tensor =>
          arrayValue.addTensor(buildBigDLTensor(tensor, attrBuilder))
          attrBuilder.clear()
        }
        attrBuilder.setDataType(DataType.ARRAY_VALUE)
        attrBuilder.setArrayValue(arrayValue)

      case tensor: Tensor[_] =>
        attrBuilder.setTensorValue(buildBigDLTensor(tensor, attrBuilder))

      case _ =>
        throw new UnsupportedOperationException("Unsupported Activity Type!")
    }
    val attr = attrBuilder.build()
    attr.toByteArray
  }

  /**
   * <h6>Deserialize Array[Byte] to activities according to `Bigdl.proto`.</h6>
   * For now, `Tensor` and `Table[Int, Tensor]` are supported.
   * It will convert `AttrValue(Array(BigdlTensor))` to a `Table`.
   * It will convert `AttrValue(BigdlTensor) ` to a `Tensor`.
   *
   * @param bytes bytes data for Activity to be deserialized
   */
  def deSerializeActivity(bytes: Array[Byte]): Activity = {
    val attr = AttrValue.parseFrom(bytes)
    attr.getDataType match {
      case DataType.ARRAY_VALUE =>
        val dataType = attr.getArrayValue.getTensor(0).getDatatype
        val tensors = getAttr(dataType, attr).asInstanceOf[Array[Tensor[_]]]
        T.array(tensors)
      case DataType.TENSOR =>
        val tValue = attr.getTensorValue
        val tensor = getAttr(tValue.getDatatype, attr)
        tensor.asInstanceOf[Tensor[_]]
      case tpe =>
        throw new UnsupportedOperationException(s"Unsupported DataType($tpe)!")
    }
  }

  private def buildBigDLTensor(tensor: Tensor[_], attrBuilder: AttrValue.Builder): BigDLTensor = {
    val status = mutable.HashMap[Int, Any]()

    val partial = partialSetAttr(tensor.getTensorNumeric(), status)
    partial(attrBuilder, tensor, ModuleSerializer.tensorType)

    val tensorId = System.identityHashCode(tensor)
    val _tensor = status(tensorId).asInstanceOf[BigDLTensor]
    val tensorBuilder = BigDLTensor.newBuilder(_tensor)

    val storageId = System.identityHashCode(tensor.storage().array())
    val _storage = status(storageId).asInstanceOf[TensorStorage]
    tensorBuilder.setStorage(_storage)

    tensorBuilder.build()
  }

  private def partialSetAttr(numeric: TensorNumeric[_], status: mutable.HashMap[Int, Any]) = {
    numeric match {
      case NumericFloat =>
        val sc = SerializeContext[Float](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[Float](sc, attrBuilder, value, tpe)
      case NumericDouble =>
        val sc = SerializeContext[Double](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[Double](sc, attrBuilder, value, tpe)
      case NumericChar =>
        val sc = SerializeContext[Char](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[Char](sc, attrBuilder, value, tpe)
      case NumericBoolean =>
        val sc = SerializeContext[Boolean](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[Boolean](sc, attrBuilder, value, tpe)
      case NumericString =>
        val sc = SerializeContext[String](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[String](sc, attrBuilder, value, tpe)
      case NumericInt =>
        val sc = SerializeContext[Int](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[Int](sc, attrBuilder, value, tpe)
      case NumericLong =>
        val sc = SerializeContext[Long](null, status, ProtoStorageType)
        (attrBuilder: AttrValue.Builder, value: Any, tpe: Type) =>
          DataConverter.setAttributeValue[Long](sc, attrBuilder, value, tpe)
    }
  }

  private def getAttr(dataType: DataType, attr: AttrValue) = {
    val status = mutable.HashMap[Int, Any]()
    val dsc = DeserializeContext(null, status, ProtoStorageType)
    dataType match {
      case DataType.INT32 =>
        DataConverter.getAttributeValue[Int](dsc, attr)
      case DataType.INT64 =>
        DataConverter.getAttributeValue[Long](dsc, attr)
      case DataType.FLOAT =>
        DataConverter.getAttributeValue[Float](dsc, attr)
      case DataType.DOUBLE =>
        DataConverter.getAttributeValue[Double](dsc, attr)
      case DataType.STRING =>
        DataConverter.getAttributeValue[String](dsc, attr)
      case DataType.BOOL =>
        DataConverter.getAttributeValue[Boolean](dsc, attr)
      case DataType.CHAR =>
        DataConverter.getAttributeValue[Char](dsc, attr)
    }
  }

}
