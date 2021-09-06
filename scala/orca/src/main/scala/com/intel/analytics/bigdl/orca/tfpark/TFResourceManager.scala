/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.tfpark

import java.nio._

import com.intel.analytics.bigdl.tensor.Tensor
import org.tensorflow.types.UInt8
import org.tensorflow.{DataType, Tensor => TTensor}

import scala.collection.mutable


class TFResourceManager() extends java.io.Serializable {
  private val tensorList: mutable.Set[TTensor[_]] = mutable.Set[TTensor[_]]()
  def createTFTensor(shape: Array[Long], buffer: FloatBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    tensorList += TFTensor
    TFTensor
  }
  def createTFTensor(shape: Array[Long], buffer: ByteBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(classOf[UInt8], shape, buffer)
    tensorList += TFTensor
    TFTensor
  }
  def createTFTensor(shape: Array[Long], buffer: IntBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    tensorList += TFTensor
    TFTensor
  }
  def createTFTensor(shape: Array[Long], buffer: LongBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    tensorList += TFTensor
    TFTensor
  }
  def createTFTensor(shape: Array[Long], buffer: DoubleBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    tensorList += TFTensor
    TFTensor
  }

  def createBoolTFTensor(shape: Array[Long], bytes: ByteBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(classOf[java.lang.Boolean], shape, bytes)
    tensorList += TFTensor
    TFTensor
  }

  def createStringTFTensor(data: Array[String]): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(data.map(_.getBytes("UTF-8")))
    tensorList += TFTensor
    TFTensor
  }

  def createStringTFTensor(data: Array[Array[Byte]]): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(data)
    tensorList += TFTensor
    TFTensor
  }

  def releaseTensor(t: TTensor[_]): Unit = {
    t.close()
    tensorList -= t
  }

  def isEmpty: Boolean = {
    tensorList.isEmpty
  }

  def destructTFTensors(): Unit = {
    for (tensor <- tensorList) {
      tensor.close()
    }

    tensorList.clear()
  }


  def bigdl2Tf(t: Tensor[_], dataType: DataType): TTensor[_] = {

    require(t.isContiguous(), "input to tfnet must be contiguous")
    val shape = t.size().map(_.toLong)
    val arr = t.storage().array()
    val offset: Int = t.storageOffset() - 1
    val length: Int = shape.product.toInt

    if (dataType == DataType.FLOAT) {
      val floatArr = arr.asInstanceOf[Array[Float]]
      val buffer = FloatBuffer.wrap(floatArr, offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.UINT8) {
      val floatArr = arr.asInstanceOf[Array[Float]]
      val buffer = ByteBuffer.wrap(floatToUint8(floatArr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT32) {
      val floatArr = arr.asInstanceOf[Array[Float]]
      val buffer = IntBuffer.wrap(floatToInt(floatArr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT64) {
      val floatArr = arr.asInstanceOf[Array[Float]]
      val buffer = LongBuffer.wrap(floatToLong(floatArr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.DOUBLE) {
      val floatArr = arr.asInstanceOf[Array[Float]]
      val buffer = DoubleBuffer.wrap(floatToDouble(floatArr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.BOOL) {
      val floatArr = arr.asInstanceOf[Array[Float]]
      val buffer = ByteBuffer.wrap(floatToBool(floatArr), offset, length)
      createBoolTFTensor(shape, buffer)
    } else if (dataType == DataType.STRING) {
      require(shape.length <= 1)
      arr match {
        case a: Array[String] =>
          createStringTFTensor(a.slice(offset, offset + length))
        case a: Array[Array[Byte]] =>
          createStringTFTensor(a.slice(offset, offset + length))
        case _ => throw new IllegalArgumentException("Analytics Zoo Tensor type must be" +
          "String or Array[Byte] to feed a TF String Tensor")
      }
    } else {
      throw new Exception(s"data type ${dataType} are not supported")
    }
  }

  private def floatToInt(array: Array[Float]): Array[Int] = {
    val result = new Array[Int](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toInt
      i = i + 1
    }
    result
  }

  private def floatToLong(array: Array[Float]): Array[Long] = {
    val result = new Array[Long](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toLong
      i = i + 1
    }
    result
  }

  private def floatToDouble(array: Array[Float]): Array[Double] = {
    val result = new Array[Double](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toDouble
      i = i + 1
    }
    result
  }

  private def floatToUint8(array: Array[Float]): Array[Byte] = {
    val result = new Array[Byte](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toByte
      i = i + 1
    }
    result
  }

  private def floatToBool(array: Array[Float]): Array[Byte] = {
    val result = new Array[Byte](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = if (array(i) == 0.0) 0.toByte else 1.toByte
      i = i + 1
    }
    result
  }


  def tensor2TFTensors(input: Seq[Tensor[_]], types: Seq[DataType],
                               tfTensors: Array[TTensor[_]]): Unit = {
    val t = input
    require(tfTensors.length == t.length, "activity and tfTensors size does not equal," +
      s" activity length is ${t.length} tfTensors length is ${tfTensors.length}")
    var i = 0
    while (i < t.length) {
      val tfTensor = bigdl2Tf(t(i), types(i))
      if (tfTensors(i) != null) {
        tfTensors(i).close()
      }
      tfTensors(i) = tfTensor
      i += 1
    }
  }
}
