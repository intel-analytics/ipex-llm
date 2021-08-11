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

package com.intel.analytics.bigdl.utils.serializer.converters

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.quantized._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{BigDLStorage, DeserializeContext, ProtoStorageType, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl._

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * DataConverter for [[com.intel.analytics.bigdl.tensor.Tensor]]
 */
object TensorConverter extends DataConverter {


  private def isEmptyTensor(tensor : Tensor[_]): Boolean = {
    val emptyTensor = tensor.getTensorType match {
      case DenseType =>
        tensor.storage == null
      case QuantizedType =>
        tensor.asInstanceOf[QuantizedTensor[_]].getStorage == null
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    emptyTensor
  }

  override def getAttributeValue[T: ClassTag](context: DeserializeContext,
                                              attribute: AttrValue)
                                             (implicit ev: TensorNumeric[T]): AnyRef = {
    val serializedTensor = attribute.getTensorValue
    if (!serializedTensor.hasStorage) {
      return null
    }
    val storages = context.storages
    val tensorId = serializedTensor.getId
    if (storages.contains(tensorId)) {
      return storages.get(tensorId).get.asInstanceOf[AnyRef]
    }
    val dataType = serializedTensor.getDatatype
    val tensorType = serializedTensor.getTensorType
    val sizes = serializedTensor.getSizeList.asScala.toArray.map(_.intValue())
    val strides = serializedTensor.getStrideList.asScala.toArray.map(_.intValue())
    val offSet = serializedTensor.getOffset
    val isScalr = serializedTensor.getIsScalar
    val serializedStorage = serializedTensor.getStorage
    val storageId = serializedStorage.getId
    val created = if (storages.contains(storageId)) {
      storages.get(storageId).get
    } else {
      null
    }

    def quant(): Tensor[T] = {
      var bytes: Array[Byte] = null
      if (context.storageType == ProtoStorageType) {
        bytes = serializedStorage.getBytesDataList.asScala.toArray.head.toByteArray
      } else {
        created
      }
      val serializedParams = serializedStorage.getIntDataList.asScala.toArray.map(_.intValue())
      val paramsNum = serializedParams.head
      val paramsArray = serializedParams.slice(1, paramsNum + 1)
      val descTypeEnum = serializedParams(1 + paramsNum)

      val start = paramsNum + 2 // params number indicator + params number + desc type

      val length = if (sizes.length == 1) {
        1 // if the size is 1, means it's a vector
      } else {
        sizes(0)
      }
      val max = new Array[T](length)
      val min = new Array[T](length)
      val sum = new Array[T](length)

      dataType match {
        case DataType.FLOAT =>
          val data = serializedStorage.getFloatDataList.asScala.toArray.map(_.floatValue())
          var i = 0
          while (i < length) {
            max(i) = ev.fromType[Float](data(i))
            min(i) = ev.fromType[Float](data(i + length))
            sum(i) = ev.fromType[Float](data(i + 2 * length))
            i += 1
          }
      }

      var params: DescParams = null

      descTypeEnum match {
        case 0 =>
          params = ConvDataParams(paramsArray)
        case 1 =>
          params = ConvWeightParams(paramsArray)
        case 2 =>
          params = LinearDataParams(paramsArray)
        case 3 =>
          params = LinearWeightParams(paramsArray)
      }

      QuantizedTensor[T](bytes, max, min, sum, sizes, params)
    }

    val tensor = dataType match {
      case DataType.FLOAT =>
        tensorType match {
          case TensorType.DENSE =>
            val storage : Storage[Float] = if (created == null ) {
              if (storageId == -1) {
                null
              } else {
                val data = serializedStorage.getFloatDataList.asScala.toArray.map(_.floatValue())
                val newStorage = Storage[Float](data)
                storages(storageId) = newStorage
                newStorage
              }
            } else created.asInstanceOf[Storage[Float]]
            Tensor[Float](storage, offSet, sizes, strides)
          case TensorType.QUANT => quant()
        }
      case DataType.DOUBLE =>
        val storage : Storage[Double] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getDoubleDataList.asScala.toArray.map(_.doubleValue())
            val newStorage = Storage[Double](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[Double]]
        Tensor[Double](storage, offSet, sizes, strides)
      case DataType.BOOL =>
        val storage : Storage[Boolean] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getBoolDataList.asScala.toArray.map(_.booleanValue())
            val newStorage = Storage[Boolean](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[Boolean]]
        Tensor[Boolean](storage, offSet, sizes, strides)
      case DataType.CHAR =>
        val storage: Storage[Char] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getIntDataList.asScala.toArray.map(_.toChar.charValue())
            val newStorage = Storage[Char](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[Char]]
        Tensor[Char](storage, offSet, sizes, strides)
      case DataType.STRING =>
        val storage: Storage[String] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getStringDataList.asScala.toArray
            val newStorage = Storage[String](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[String]]
        Tensor[String](storage, offSet, sizes, strides)
      case DataType.INT32 =>
        val storage: Storage[Int] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getIntDataList.asScala.toArray.map(_.intValue())
            val newStorage = Storage[Int](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[Int]]
        Tensor[Int](storage, offSet, sizes, strides)
      case DataType.SHORT =>
        val storage: Storage[Short] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getIntDataList.asScala.toArray.map(_.shortValue())
            val newStorage = Storage[Short](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[Short]]
        Tensor[Short](storage, offSet, sizes, strides)
      case DataType.INT64 =>
        val storage: Storage[Long] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getLongDataList.asScala.toArray.map(_.longValue())
            val newStorage = Storage[Long](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[Long]]
        Tensor[Long](storage, offSet, sizes, strides)
      case DataType.BYTES =>
        import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
        val storage: Storage[ByteString] = if (created == null ) {
          if (storageId == -1) {
            null
          } else {
            val data = serializedStorage.getBytesDataList.asScala.toArray
            val newStorage = Storage[ByteString](data)
            storages(storageId) = newStorage
            newStorage
          }
        } else created.asInstanceOf[Storage[ByteString]]
        Tensor[ByteString](storage, offSet, sizes, strides)
      case _ => throw new IllegalArgumentException(s"$dataType not supported in tensor now !")
    }
    storages(tensorId) = tensor
    tensor
  }

  private def setStorage[T: ClassTag](context: SerializeContext[T],
    tensorBuilder: BigDLTensor.Builder, tensor: Tensor[_]): Unit = {
    val storageType = context.storageType
    if (storageType == ProtoStorageType) {
      ProtoTensorStorageManager.setStorage(context, tensorBuilder, tensor)
    } else if (storageType == BigDLStorage) {
      BigDLTensorStorageManager.setStorage(context, tensorBuilder, tensor)
    } else {
      throw new IllegalArgumentException(s"$storageType not supported")
    }
  }

  override def setAttributeValue[T: ClassTag]
  (context: SerializeContext[T], attributeBuilder: AttrValue.Builder, value: Any,
   valueType : universe.Type = null)
  (implicit ev: TensorNumeric[T]): Unit = {
    attributeBuilder.setDataType(DataType.TENSOR)
    if (value != null) {
      val tensor = value.asInstanceOf[Tensor[_]]
      val tensorId = System.identityHashCode(tensor)
      val storages = context.storages
      // Check if tensor has been shared
      if (storages.contains(tensorId)) {
        attributeBuilder.setTensorValue(resetTensor(storages.get(tensorId).get
          .asInstanceOf[BigDLTensor]))
      } else {
        val totalElement = tensor.nElement()
        val dimension = tensor.dim()
        val tensorBuilder = BigDLTensor.newBuilder
        tensorBuilder.setId(tensorId)
        tensorBuilder.setDimension(dimension)
        tensorBuilder.setNElements(totalElement)
        tensor.getTensorType match {
          case DenseType =>
            tensorBuilder.setOffset(tensor.storageOffset())
            tensorBuilder.setIsScalar(tensor.isScalar)
            tensorBuilder.setTensorType(TensorType.DENSE)
          case QuantizedType =>
            tensorBuilder.setTensorType(TensorType.QUANT)
          case t => throw new NotImplementedError(s"$t is not supported")
        }

        val tensorEmpty = isEmptyTensor(tensor)

        if (!tensorEmpty) {
          tensor.size().foreach(size => tensorBuilder.addSize(size))
          tensor.stride().foreach(stride => tensorBuilder.addStride(stride))
        }
        setStorage(context, tensorBuilder, tensor)
        val tensorBuild = tensorBuilder.build
        attributeBuilder.setTensorValue(resetTensor(tensorBuild))
        storages(tensorId) = tensorBuild
      }
    }
  }


  private def resetStorage(originStorage : TensorStorage) : TensorStorage = {
    val storageBuilder = TensorStorage.newBuilder
    storageBuilder.setDatatype(originStorage.getDatatype)
    storageBuilder.setId(originStorage.getId)
    storageBuilder.build
  }

  private def resetTensor(originTensor: BigDLTensor) : BigDLTensor = {
    val tensorBuilder = BigDLTensor.newBuilder(originTensor)
    tensorBuilder.clearStorage
    tensorBuilder.setDatatype(originTensor.getDatatype)
    tensorBuilder.setId(originTensor.getId)
    if (originTensor.hasStorage) {
      tensorBuilder.setStorage(resetStorage(originTensor.getStorage))
    }
    tensorBuilder.build
  }
}
