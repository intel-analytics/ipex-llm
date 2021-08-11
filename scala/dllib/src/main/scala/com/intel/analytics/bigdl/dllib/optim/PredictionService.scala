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

import java.util.concurrent.LinkedBlockingQueue

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.serialization.Bigdl.AttrValue.ArrayValue
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLTensor, DataType, TensorStorage}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericChar, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericString}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializer, ProtoStorageType, SerializeContext}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.Type
import scala.util.{Failure, Success, Try}

/**
 * <h6>Thread-safe Prediction Service for Concurrent Calls</h6>
 * In this service, concurrency is kept not greater than [[numThreads]] by a `BlockingQueue`,
 * which contains available model instances.
 * <br><br>
 * [[numThreads]] model instances sharing weights/bias
 * will be put into the `BlockingQueue` during initialization.
 * <br><br>
 * When predict method called, service will try to take an instance from `BlockingQueue`,
 * which means if all instances are on serving, the predicting request will be blocked until
 * some instances are released.
 * <br><br>
 * If exceptions caught during predict,
 * a scalar Tensor[String] will be returned with thrown message.
 *
 * @param model BigDL model used to do predictions
 * @param numThreads max concurrency
 */
class PredictionService[T: ClassTag] private[optim](
    model: Module[T],
    numThreads: Int
)(implicit ev: TensorNumeric[T]) {

  protected val instQueue: LinkedBlockingQueue[Module[T]] = {
    val shallowCopies = (1 to numThreads)
      .map(_ => model.clone(false).evaluate()).asJava

    new LinkedBlockingQueue[Module[T]](shallowCopies)
  }

  /**
   * <h6>Thread-safe single sample prediction</h6>
   * Running model prediction with input Activity as soon as
   * there exists vacant instances(the size of pool is [[numThreads]]).
   * Otherwise, it will hold on till some instances are released.
   * <br><br>
   * Outputs will be deeply copied after model prediction, so they are invariant.
   *
   * @param request input Activity, could be Tensor or Table(key, Tensor)
   * @return output Activity, could be Tensor or Table(key, Tensor)
   */
  def predict(request: Activity): Activity = {
    // Take an instance from blocking queue,
    // it will cause a thread blocking when no instance is available.
    val module = instQueue.take()

    // do predictions
    val forwardResult = Try(module.forward(request)) match {
      case Success(activity) => activity
      case Failure(e) => errorTensor("running forward", e)
    }

    // cloned values after prediction finished
    val output = try {
      forwardResult match {
        case tensor: Tensor[_] =>
          tensor.clone()
        case table: Table =>
          val clonedMap = mutable.HashMap[Any, Any]()
          table.getState().foreach { x => (x: @unchecked) match {
            case (k: Tensor[_], v: Tensor[_]) =>
              clonedMap += k.clone() -> v.clone()
            case (k, v: Tensor[_]) =>
              clonedMap += k -> v.clone()
            }
          }
          new Table(clonedMap)
      }
    } catch {
      case e: Throwable => errorTensor("Clone Result", e)
    } finally {
      // Release module instance back to blocking queue
      instQueue.offer(module)
    }

    output
  }

  /**
   * <h6>Thread-safe single sample prediction</h6>
   * Firstly, deserialization tasks will be run with inputs(Array[Byte]).
   * <br><br>
   * Then, run model prediction with deserialized inputs
   * as soon as there exists vacant instances(total number is [[numThreads]]).
   * Otherwise, it will hold on till some instances are released.
   * <br><br>
   * Finally, prediction results will be serialized to Array[Byte] according to BigDL.proto.
   *
   * @param request input bytes, which will be deserialized by BigDL.proto
   * @return output bytes, which is serialized by BigDl.proto
   */
  def predict(request: Array[Byte]): Array[Byte] = {
    val output = Try(
      PredictionService.deSerializeActivity(request)
    ) match {
      case Success(activity) => predict(activity)
      case Failure(e) => errorTensor("DeSerialize Input", e)
    }

    val bytesOut = try {
      PredictionService.serializeActivity(output)
    } catch {
      case e: Throwable =>
        val act = errorTensor("Serialize Output", e)
        PredictionService.serializeActivity(act)
    }

    bytesOut
  }

  private def errorTensor(stage: String, e: Throwable): Tensor[String] = {
    val msg = s"Exception caught during [$stage]! \n" +
      s"The message is ${e.getMessage} \n" +
      s"The cause is ${e.getCause}"
    Tensor.scalar(msg)
  }

}


object PredictionService {

  /**
   * <h6>Thread-safe Prediction Service for Concurrent Calls</h6>
   * In this service, concurrency is kept not greater than `numThreads` by a `BlockingQueue`,
   * which contains available model instances.
   * <br><br>
   * If exceptions caught during predict,
   * a scalar Tensor[String] will be returned with thrown message.
   *
   * @param model BigDL model used to do predictions
   * @param numThreads max concurrency
   * @return a PredictionService instance
   */
  def apply[T: ClassTag](
      model: Module[T],
      numThreads: Int
  )(implicit ev: TensorNumeric[T]): PredictionService[T] = {
    new PredictionService[T](model, numThreads)
  }

  /**
   * <h6>Serialize activities to Array[Byte] according to `Bigdl.proto`.</h6>
   * For now, `Tensor` and `Table[primitive|Tensor, Tensor]` are supported.
   *
   * @param activity activity to be serialized
   */
  def serializeActivity(activity: Activity): Array[Byte] = {
    val attrBuilder = AttrValue.newBuilder()
    activity match {
      case table: Table =>
        var keyIsPrimitive = true
        val firstKey = table.getState().head._1
        val tensorState: Array[(Tensor[_], Tensor[_])] = firstKey match {
          case _: Tensor[_] =>
            keyIsPrimitive = false
            table.getState().map { x => (x: @unchecked) match { case (k: Tensor[_], v: Tensor[_]) =>
              k -> v }}.toArray
          case _: Int =>
            table.getState().map { x => (x: @unchecked) match { case (k: Int, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: Long =>
            table.getState().map { x => (x: @unchecked) match { case (k: Long, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: Char =>
            table.getState().map { x => (x: @unchecked) match { case (k: Char, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: Short =>
            table.getState().map {x => (x: @unchecked) match { case (k: Short, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: Float =>
            table.getState().map { x => (x: @unchecked) match { case (k: Float, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: Double =>
            table.getState().map { x => (x: @unchecked) match { case (k: Double, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: Boolean =>
            table.getState().map { x => (x: @unchecked) match { case (k: Boolean, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case _: String =>
            table.getState().map { x => (x: @unchecked) match { case (k: String, v: Tensor[_]) =>
              Tensor.scalar(k) -> v }}.toArray
          case key =>
            throw new UnsupportedOperationException(s"Unsupported Table key: $key!")
        }

        val (keys, values) = tensorState.unzip
        // tensors structure: [isKeyPrimitive, keys, values]
        val tensors = Array(Tensor.scalar(keyIsPrimitive)) ++ keys ++ values

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
   * For now, `Tensor` and `Table[primitive|Tensor, Tensor]` are supported.
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
        // tensors structure: [isKeyPrimitive, keys, values]
        val tensors = getAttr(dataType, attr).asInstanceOf[Array[Tensor[_]]]

        val nElement = (tensors.length - 1) / 2
        val keyIsPrimitive = tensors.head.asInstanceOf[Tensor[Boolean]].value()
        val _keys = tensors.slice(1, nElement + 1)
        val keys = if (keyIsPrimitive) _keys.map(_.value()) else _keys
        val values = tensors.slice(nElement + 1, tensors.length)
        val table = T()
        keys.zip(values).foreach { case(k, v) => table.update(k, v) }
        table

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
      case _ => throw new UnsupportedOperationException(s"Unsupported DataType($dataType)!")
    }
  }

}
