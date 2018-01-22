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

import java.io.{DataInputStream, DataOutputStream}

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Storage
import com.intel.analytics.bigdl.utils.serializer.BigDLDataType
import com.intel.analytics.bigdl.utils.serializer.BigDLDataType.BigDLDataType

/**
 * DataReaderWriter defines how to read/write weight data from bin file
 */
trait DataReaderWriter {
  def write(outputStream: DataOutputStream, data: Array[_]): Unit
  def read(inputStream: DataInputStream, size: Int): Any
  def dataType(): BigDLDataType
}

object FloatReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeFloat(d.asInstanceOf[Float]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Float](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readFloat
    }
    Storage[Float](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.FLOAT
}

object DoubleReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeDouble(d.asInstanceOf[Double]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Double](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readDouble
    }
    Storage[Double](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.DOUBLE
}

object CharReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeChar(d.asInstanceOf[Char]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Char](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readChar
    }
    Storage[Char](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.CHAR
}

object BoolReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeBoolean(d.asInstanceOf[Boolean]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Boolean](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readBoolean
    }
    Storage[Boolean](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.BOOL
}

object StringReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(str => {
      val value = str.asInstanceOf[String].getBytes("utf-8")
      outputStream.writeInt(value.size)
      outputStream.write(value)
    })
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[String](size)
    for (i <- 0 until size) {
      val ssize = inputStream.readInt
      val buffer = new Array[Byte](ssize)
      inputStream.read(buffer)
      data(i) = new String(buffer, "utf-8")
    }
    Storage[String](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.STRING
}

object IntReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeInt(d.asInstanceOf[Int]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Int](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readInt
    }
    Storage[Int](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.INT
}

object ShortReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeShort(d.asInstanceOf[Short]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Short](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readShort
    }
    Storage[Short](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.SHORT
}

object LongReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(d => outputStream.writeLong(d.asInstanceOf[Long]))
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Long](size)
    for (i <- 0 until size) {
      data(i) = inputStream.readLong
    }
    Storage[Long](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.LONG
}

object ByteStringReaderWriter extends DataReaderWriter {
  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
    data.foreach(str => {
      val value = str.asInstanceOf[ByteString].toByteArray
      outputStream.writeInt(value.size)
      outputStream.write(value)
    })
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[ByteString](size)
    for (i <- 0 until size) {
      val ssize = inputStream.readInt
      val buffer = new Array[Byte](ssize)
      inputStream.read(buffer)
      data(i) = ByteString.copyFrom(buffer)
    }
    Storage[ByteString](data)
  }

  def dataType(): BigDLDataType = BigDLDataType.BYTESTRING
}

object ByteReaderWriter extends DataReaderWriter {

  override def write(outputStream: DataOutputStream, data: Array[_]): Unit = {
   outputStream.write(data.asInstanceOf[Array[Byte]])
  }

  override def read(inputStream: DataInputStream, size: Int): Any = {
    val data = new Array[Byte](size)
    inputStream.read(data)
    Storage[Byte](data)
  }

  override def dataType(): BigDLDataType = BigDLDataType.BYTE
}

object DataReaderWriter {
  def apply(datas : Array[_]): DataReaderWriter = {
    datas match {
      case flats: Array[Float] => FloatReaderWriter
      case dbls: Array[Double] => DoubleReaderWriter
      case chs: Array[Char] => CharReaderWriter
      case bools: Array[Boolean] => BoolReaderWriter
      case strs : Array[String] => StringReaderWriter
      case ints : Array[Int] => IntReaderWriter
      case shorts : Array[Short] => ShortReaderWriter
      case longs : Array[Long] => LongReaderWriter
      case bytestrs : Array[ByteString] => ByteStringReaderWriter
      case bytes : Array[Byte] => ByteReaderWriter
      case _ => throw new RuntimeException("Unsupported Type")
    }
  }

  def apply(dataType : BigDLDataType): DataReaderWriter = {
    dataType match {
      case BigDLDataType.FLOAT => FloatReaderWriter
      case BigDLDataType.DOUBLE => DoubleReaderWriter
      case BigDLDataType.CHAR => CharReaderWriter
      case BigDLDataType.BOOL => BoolReaderWriter
      case BigDLDataType.STRING => StringReaderWriter
      case BigDLDataType.INT => IntReaderWriter
      case BigDLDataType.SHORT => ShortReaderWriter
      case BigDLDataType.LONG => LongReaderWriter
      case BigDLDataType.BYTESTRING => ByteStringReaderWriter
      case BigDLDataType.BYTE => ByteReaderWriter
      case _ => throw new RuntimeException("Unsupported Type")
    }
  }
}
