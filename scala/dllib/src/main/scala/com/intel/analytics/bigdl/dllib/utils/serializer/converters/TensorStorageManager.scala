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
import com.intel.analytics.bigdl.nn.quantized.{ConvData, ConvWeight, LinearData, LinearWeight}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.{NumericBoolean, NumericChar, NumericDouble, NumericFloat, NumericInt, NumericLong, NumericShort, NumericString}
import com.intel.analytics.bigdl.tensor.{DenseType, QuantizedTensor, QuantizedType, Tensor}
import com.intel.analytics.bigdl.utils.serializer.SerializeContext
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
import com.intel.analytics.bigdl.serialization.Bigdl.{BigDLTensor, DataType, TensorStorage}

import scala.reflect.ClassTag


trait TensorStorageManager {
  def setStorage[T: ClassTag](context: SerializeContext[T],
    tensorBuilder: BigDLTensor.Builder, tensor: Tensor[_]): Unit

  protected def isEmptyTensor(tensor : Tensor[_]): Boolean = {
    val emptyTensor = tensor.getTensorType match {
      case DenseType =>
        tensor.storage == null
      case QuantizedType =>
        tensor.asInstanceOf[QuantizedTensor[_]].getStorage == null
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    emptyTensor
  }

  protected def getStorageId[T: ClassTag](tensor: Tensor[_]): Int = {
    val isEmpty = isEmptyTensor(tensor)
    tensor.getTensorType match {
      case DenseType =>
        if (isEmpty) -1 else System.identityHashCode(tensor.storage().array())
      case QuantizedType =>
        if (isEmpty) {
          -1
        } else {
          System.identityHashCode(tensor.asInstanceOf[QuantizedTensor[T]].getStorage)
        }
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }

  protected def resetStorage(originStorage : TensorStorage) : TensorStorage = {
    val storageBuilder = TensorStorage.newBuilder
    storageBuilder.setDatatype(originStorage.getDatatype)
    storageBuilder.setId(originStorage.getId)
    storageBuilder.build
  }
}

object BigDLTensorStorageManager extends TensorStorageManager {
  override def setStorage[T: ClassTag](context: SerializeContext[T],
    tensorBuilder: BigDLTensor.Builder, tensor: Tensor[_]): Unit = {
    val tensorNumeric = tensor.getTensorNumeric()
    val storageId = getStorageId(tensor)
    val storages = context.storages
    val storageBuilder = TensorStorage.newBuilder
    storageBuilder.setId(storageId)
    if (tensorNumeric == NumericFloat) {
      tensorBuilder.setDatatype(DataType.FLOAT)
      storageBuilder.setDatatype(DataType.FLOAT)
      if(tensor.getTensorType == QuantizedType) {
        val quantTensor = tensor.asInstanceOf[QuantizedTensor[Float]]
        val bytes = quantTensor.getStorage
        val bs = ByteString.copyFrom(bytes)
        storageBuilder.addBytesData(bs)

        // max, min, and sum
        quantTensor.maxOfRow.foreach(data => storageBuilder.addFloatData(data))
        quantTensor.minOfRow.foreach(data => storageBuilder.addFloatData(data))
        quantTensor.sumOfRow.foreach(data => storageBuilder.addFloatData(data))

        // params and desc type
        val params = quantTensor.params.array
        storageBuilder.addIntData(params.length)
        params.foreach(param => storageBuilder.addIntData(param.asInstanceOf[Int]))

        quantTensor.params.getType match {
          case ConvData => storageBuilder.addIntData(0)
          case ConvWeight => storageBuilder.addIntData(1)
          case LinearData => storageBuilder.addIntData(2)
          case LinearWeight => storageBuilder.addIntData(3)
        }
      }
    } else if (tensorNumeric == NumericDouble) {
      tensorBuilder.setDatatype(DataType.DOUBLE)
      storageBuilder.setDatatype(DataType.DOUBLE)
    } else if (tensorNumeric == NumericChar) {
      tensorBuilder.setDatatype(DataType.CHAR)
      storageBuilder.setDatatype(DataType.CHAR)
    } else if (tensorNumeric == NumericBoolean) {
      tensorBuilder.setDatatype(DataType.BOOL)
      storageBuilder.setDatatype(DataType.BOOL)
    } else if (tensorNumeric == NumericString) {
      tensorBuilder.setDatatype(DataType.STRING)
      storageBuilder.setDatatype(DataType.STRING)
    } else if (tensorNumeric == NumericInt) {
      tensorBuilder.setDatatype(DataType.INT32)
      storageBuilder.setDatatype(DataType.INT32)
    } else if (tensorNumeric == NumericShort) {
      tensorBuilder.setDatatype(DataType.SHORT)
      storageBuilder.setDatatype(DataType.SHORT)
    } else if (tensorNumeric == NumericLong) {
      tensorBuilder.setDatatype(DataType.INT64)
      storageBuilder.setDatatype(DataType.INT64)
    } else if (tensorNumeric == NumericByteString) {
      tensorBuilder.setDatatype(DataType.BYTES)
      storageBuilder.setDatatype(DataType.BYTES)
    }

    val storage = tensor.getTensorType match {
      case DenseType =>
        if (tensor.storage() == null) null else tensor.storage().array()
      case QuantizedType =>
        tensor.asInstanceOf[QuantizedTensor[Float]].getStorage
      case t => throw new NotImplementedError(s"$t is not supported")
    }

    if (storage != null) {
      storages(storageId) = storage
    }

    tensorBuilder.setStorage(storageBuilder.build())
  }
}

object ProtoTensorStorageManager extends TensorStorageManager {

  override def setStorage[T: ClassTag]
  (context: SerializeContext[T], tensorBuilder: BigDLTensor.Builder, tensor: Tensor[_]): Unit = {
    val tensorNumeric = tensor.getTensorNumeric()
    val isEmpty = isEmptyTensor(tensor)
    val storageId = getStorageId(tensor)
    val storages = context.storages
    if (storages.contains(storageId)) {
      val storage = storages(storageId).asInstanceOf[TensorStorage]
      tensorBuilder.setStorage(resetStorage(storage))
      // we should set back the datatype from existed storage
      tensorBuilder.setDatatype(storage.getDatatype)
    } else {
      val storageBuilder = TensorStorage.newBuilder
      if (tensorNumeric == NumericFloat) {
        tensorBuilder.setDatatype(DataType.FLOAT)
        storageBuilder.setDatatype(DataType.FLOAT)
        if(!isEmpty) {
          tensor.getTensorType match {
            case DenseType =>
              tensor.storage().array().asInstanceOf[Array[Float]].
                foreach(data => storageBuilder.addFloatData(data))
            case QuantizedType =>
              val quantTensor = tensor.asInstanceOf[QuantizedTensor[Float]]
              val bytes = quantTensor.getStorage
              val bs = ByteString.copyFrom(bytes)
              storageBuilder.addBytesData(bs)

              // max, min, and sum
              quantTensor.maxOfRow.foreach(data => storageBuilder.addFloatData(data))
              quantTensor.minOfRow.foreach(data => storageBuilder.addFloatData(data))
              quantTensor.sumOfRow.foreach(data => storageBuilder.addFloatData(data))

              // params and desc type
              val params = quantTensor.params.array
              storageBuilder.addIntData(params.length)
              params.foreach(param => storageBuilder.addIntData(param.asInstanceOf[Int]))

              quantTensor.params.getType match {
                case ConvData => storageBuilder.addIntData(0)
                case ConvWeight => storageBuilder.addIntData(1)
                case LinearData => storageBuilder.addIntData(2)
                case LinearWeight => storageBuilder.addIntData(3)
              }
            case t => throw new NotImplementedError(s"$t is not supported")
          }
        }
      } else if (tensorNumeric == NumericDouble) {
        tensorBuilder.setDatatype(DataType.DOUBLE)
        storageBuilder.setDatatype(DataType.DOUBLE)
        if(!tensor.isEmpty) {
          tensor.storage().array().asInstanceOf[Array[Double]].
            foreach(data => storageBuilder.addDoubleData(data))
        }
      } else if (tensorNumeric == NumericChar) {
        tensorBuilder.setDatatype(DataType.CHAR)
        storageBuilder.setDatatype(DataType.CHAR)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[Char]].
            foreach(data => storageBuilder.addIntData(data))
        }
      } else if (tensorNumeric == NumericBoolean) {
        tensorBuilder.setDatatype(DataType.BOOL)
        storageBuilder.setDatatype(DataType.BOOL)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[Boolean]].
            foreach(data => storageBuilder.addBoolData(data))
        }
      } else if (tensorNumeric == NumericString) {
        tensorBuilder.setDatatype(DataType.STRING)
        storageBuilder.setDatatype(DataType.STRING)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[String]].
            foreach(data => storageBuilder.addStringData(data))
        }
      } else if (tensorNumeric == NumericInt) {
        tensorBuilder.setDatatype(DataType.INT32)
        storageBuilder.setDatatype(DataType.INT32)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[Int]].
            foreach(data => storageBuilder.addIntData(data))
        }
      } else if (tensorNumeric == NumericShort) {
        tensorBuilder.setDatatype(DataType.SHORT)
        storageBuilder.setDatatype(DataType.SHORT)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[Short]].
            foreach(data => storageBuilder.addIntData(data))
        }
      } else if (tensorNumeric == NumericLong) {
        tensorBuilder.setDatatype(DataType.INT64)
        storageBuilder.setDatatype(DataType.INT64)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[Long]].
            foreach(data => storageBuilder.addLongData(data))
        }
      } else if (tensorNumeric == NumericByteString) {
        tensorBuilder.setDatatype(DataType.BYTES)
        storageBuilder.setDatatype(DataType.BYTES)
        if(!isEmpty) {
          tensor.storage().array().asInstanceOf[Array[ByteString]].
            foreach(data => storageBuilder.addBytesData(data))
        }
      }
      storageBuilder.setId(storageId)
      val storage = storageBuilder.build
      tensorBuilder.setStorage(resetStorage(storage))
      if (storageId != -1) {
        storages(storageId) = storage
      }
    }
  }
}
