/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.utils

import java.io._
import java.nio._
import java.nio.file._
import java.util.{HashMap, Map}

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}


sealed abstract class TorchObject(val typeId: Int)

object TorchObject {

  case object TYPE_NIL extends TorchObject(0)

  case object TYPE_NUMBER extends TorchObject(1)

  case object TYPE_FLOAT_TENSOR extends TorchObject(4)

  case object TYPE_DOUBLE_TENSOR extends TorchObject(4)

  case object TYPE_DOUBLE_STORAGE extends TorchObject(4)

  case object TYPE_FLOAT_STORAGE extends TorchObject(4)

  case object TYPE_LONG_STORAGE extends TorchObject(4)

  case object TYPE_LINEAR extends TorchObject(4)

  case object TYPE_STRING extends TorchObject(2)

  case object TYPE_BOOLEAN extends TorchObject(5)

  case object TYPE_TABLE extends TorchObject(3)

  case object TYPE_SPATIALCONVOLUTION extends TorchObject(4)

  case object TYPE_SPATIALMAXPOOLING extends TorchObject(4)

  case object TYPE_THRESHOLD extends TorchObject(4)

  case object TYPE_CONCAT extends TorchObject(4)

  case object TYPE_SEQUENTIAL extends TorchObject(4)

  case object TYPE_VIEW extends TorchObject(4)

  case object TYPE_DROPOUT extends TorchObject(4)

}

object File {

  import TorchObject._

  var i = 0

  /**
   * Load torch object from a torch format file
   *
   * @param fileName
   * @tparam T
   * @return
   */
  def load[T](fileName: String): T = {
    val path = Paths.get(fileName)
    val rawData = ByteBuffer.wrap(Files.readAllBytes(path))
    rawData.order(ByteOrder.LITTLE_ENDIAN)
    val objects: Map[Int, Any] = new HashMap()
    readObject(rawData, objects).asInstanceOf[T]
  }

  /**
   * Save torch object into a torch format file
   *
   * @param source
   * @param fileName
   * @param objectType
   */
  def save(source: Any, fileName: String, objectType: TorchObject): Unit = {
    val capacity = 300
    val path = Paths.get(fileName)
    val buffer = ByteBuffer.allocate(capacity)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    writeObject(source: Any, buffer, path, objectType)
  }

  /**
   * Save torch object into a Java object file
   *
   * @param obj
   */
  def save(obj: Serializable, fileName: String, isOverwrite: Boolean = false): Unit = {
    if (Files.exists(Paths.get(fileName)) && !isOverwrite) {
      throw new RuntimeException("file exists!")
    }

    val objFile = new ObjectOutputStream(new FileOutputStream(fileName))
    objFile.writeObject(obj)
  }

  /**
   * Load torch object from a Java object file
   *
   * @param fileName
   * @tparam T
   */
  def loadObj[T](fileName: String): T = {
    val objFile = new ObjectInputStream(new FileInputStream(fileName))
    objFile.readObject().asInstanceOf[T]
  }

  def flush(rawdata: ByteBuffer, path: Path): Unit = {
    if ((rawdata.capacity() - rawdata.position()) < 8) {
      byteWrite(rawdata, path)
    }
  }

  def byteWrite(rawdata: ByteBuffer, path: Path): Unit = {
    Files.write(path, rawdata.array().slice(0, rawdata.position()), StandardOpenOption.APPEND)
    rawdata.clear()
  }


  private def readObject(rawData: ByteBuffer, objects: Map[Int, Any]): Any = {
    val TYPE_NIL: Int = 0
    val TYPE_NUMBER: Int = 1
    val TYPE_STRING: Int = 2
    val TYPE_TABLE: Int = 3
    val TYPE_TORCH: Int = 4
    val TYPE_BOOLEAN: Int = 5
    val TYPE_FUNCTION: Int = 6
    val TYPE_RECUR_FUNCTION: Int = 8
    val LEGACY_TYPE_RECUR_FUNCTION: Int = 7

    val typeId = rawData.getInt()

    typeId match {
      case TYPE_NIL => null
      case TYPE_TORCH =>
        val indexId = rawData.getInt()
        if (objects.containsKey(indexId)) {
          objects.get(indexId)
        } else {
          val (versionNumber, className) = readVersionAndClass(rawData)
          // Todo: Use reflection to do this is better
          val result = className match {
            case "torch.FloatTensor" => readFloatTensor(rawData, objects)
            case "torch.DoubleTensor" => readDoubleTensor(rawData, objects)
            case "nn.Sequential" => readSequentialModule(rawData, objects)
            case "nn.SpatialConvolutionMM" => readSpatialConvolution(rawData, objects)
            case "nn.SpatialConvolution" => readSpatialConvolution(rawData, objects)
            case "nn.SpatialZeroPadding" => readSpatialZeroPadding(rawData, objects)
            case "nn.ReLU" => readReLU(rawData, objects)
            case "nn.SpatialMaxPooling" => readSpatialMaxPooling(rawData, objects)
            case "nn.SpatialAveragePooling" => readSpatialAveragePooling(rawData, objects)
            case "nn.View" => readView(rawData, objects)
            case "nn.Linear" => readLinear(rawData, objects)
            case "nn.Threshold" => readThreshold(rawData, objects)
            case "nn.LogSoftMax" => readLogSoftMax(rawData, objects)
            case "nn.Concat" => readConcat(rawData, objects)
            case "nn.Dropout" => readDropout(rawData, objects)
            case "torch.DoubleStorage" => readDoubleStorage(rawData)
            case "torch.FloatStorage" => readFloatStorage(rawData)
            case "torch.LongStorage" => readLongStorage(rawData)
            case "nn.SpatialConvolutionMap" => readSpatialConvolutionMap(rawData, objects)
            case "nn.Tanh" => readTanh(rawData, objects)
            case "nn.Reshape" => readReshape(rawData, objects)
            case "nn.BatchNormalization" => readBatchNormalization(rawData, objects)
            case "nn.SpatialBatchNormalization" => readSpatialBatchNormalization(rawData, objects)
            case _ => throw new UnsupportedOperationException(className)
          }
          objects.put(indexId, result)
          result
        }
      case TYPE_TABLE =>
        val indexId = rawData.getInt()
        if (objects.containsKey(indexId)) {
          objects.get(indexId)
        } else {
          val result = readTable(rawData, objects)
          objects.put(indexId, result)
          result
        }
      case TYPE_NUMBER => readNumber(rawData)
      case TYPE_STRING => readString(rawData)
      case TYPE_BOOLEAN => readBoolean(rawData)
      case _ => throw new UnsupportedOperationException(typeId.toString)
    }
  }

  private def writeObject(
    source: Any, rawdata: ByteBuffer, path: Path, objectType: TorchObject): Unit = {
    flush(rawdata, path)
    rawdata.putInt(objectType.typeId)


    objectType match {
      case TYPE_NIL => return
      case TYPE_FLOAT_TENSOR =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "torch.FloatTensor", rawdata, path)
        writeFloatTensor(source.asInstanceOf[Tensor[Float]], rawdata, path)
      case TYPE_DOUBLE_TENSOR =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "torch.DoubleTensor", rawdata, path)
        writeDoubleTensor(source.asInstanceOf[Tensor[Double]], rawdata, path)
      case TYPE_FLOAT_STORAGE =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "torch.FloatStorage", rawdata, path)
        writeFloatStorage(source.asInstanceOf[Tensor[Float]], rawdata, path)
      case TYPE_DOUBLE_STORAGE =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "torch.DoubleStorage", rawdata, path)
        writeDoubleStorage(source.asInstanceOf[Tensor[Double]], rawdata, path)
      case TYPE_LONG_STORAGE =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "torch.LongStorage", rawdata, path)
        writeLongStorage(source.asInstanceOf[Array[Int]], rawdata, path)
      case TYPE_NUMBER => writeNumber(source.asInstanceOf[Double], rawdata, path)
      case TYPE_STRING => writeString(source.asInstanceOf[String], rawdata, path)
      case TYPE_BOOLEAN => writeBoolean(source.asInstanceOf[Boolean], rawdata, path)
      case TYPE_LINEAR =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.Linear", rawdata, path)
        writeLinear(source.asInstanceOf[Linear[Double]], rawdata, path)
      case TYPE_SPATIALCONVOLUTION =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.SpatialConvolutionMM", rawdata, path)
        writeSpatialConvolution(source.asInstanceOf[SpatialConvolution[Double]], rawdata, path)
      case TYPE_SPATIALMAXPOOLING =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.SpatialMaxPooling", rawdata, path)
        writeSpatialMaxPooling(source.asInstanceOf[SpatialMaxPooling[Double]], rawdata, path)
      case TYPE_THRESHOLD =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.Threshold", rawdata, path)
        writeThreshold(source.asInstanceOf[Threshold[Double]], rawdata, path)
      case TYPE_CONCAT =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.Concat", rawdata, path)
        writeConcat(source.asInstanceOf[Concat[Double]], rawdata, path)
      case TYPE_SEQUENTIAL =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.Sequential", rawdata, path)
        writeSequential(source.asInstanceOf[Sequential[Double]], rawdata, path)
      case TYPE_DROPOUT =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.Dropout", rawdata, path)
        writeDropout(source.asInstanceOf[Dropout[Double]], rawdata, path)
      case TYPE_VIEW =>
        i = i + 1
        rawdata.putInt(i)
        writeVersionAndClass("V 1", "nn.View", rawdata, path)
        writeView(source.asInstanceOf[View[Double]], rawdata, path)
      case TYPE_TABLE =>
        i = i + 1
        rawdata.putInt(i)
        writeTable(source.asInstanceOf[Map[Any, Any]], rawdata, path)
      case _ => throw new UnsupportedOperationException(objectType.toString)

    }
  }

  private def writeNumber(source: Double, rawdata: ByteBuffer, path: Path): Unit = {
    flush(rawdata, path)
    rawdata.putDouble(source)
    byteWrite(rawdata, path)
  }

  private def writeVersionAndClass(version: String, className: String, rawdata: ByteBuffer,
    path: Path): Unit = {
    writeString(version, rawdata, path)
    writeString(className, rawdata, path)
  }

  private def writeString(string: String, rawdata: ByteBuffer, path: Path): Unit = {
    val length = string.length
    flush(rawdata, path)
    rawdata.putInt(length)
    var i = 0
    while (i < length) {
      flush(rawdata, path)
      rawdata.put(string(i).toByte)
      i += 1
    }
    byteWrite(rawdata, path)
  }

  private def writeBoolean(source: Boolean, rawdata: ByteBuffer, path: Path): Unit = {
    var tmp = 1
    if (source == false) {
      tmp = 0
    }
    flush(rawdata, path)
    rawdata.putInt(tmp)
    byteWrite(rawdata, path)
  }

  private def writeDoubleTensor(source: Tensor[Double], rawdata: ByteBuffer, path: Path): Unit = {
    val ndimension = source.dim()
    flush(rawdata, path)
    rawdata.putInt(ndimension)

    var i = 0
    while (i < ndimension) {
      flush(rawdata, path)
      rawdata.putLong(source.size(i + 1))
      i += 1
    }

    i = 0
    while (i < ndimension) {
      flush(rawdata, path)
      rawdata.putLong(source.stride(i + 1))
      i += 1
    }
    flush(rawdata, path)
    rawdata.putLong(source.storageOffset())

    if (ndimension == 0) {
      writeObject(source, rawdata, path, TYPE_NIL)
    }
    else {
      writeObject(source, rawdata, path, TYPE_DOUBLE_STORAGE)
    }

    byteWrite(rawdata, path)
  }

  private def writeFloatTensor(source: Tensor[Float], rawdata: ByteBuffer, path: Path): Unit = {
    val ndimension = source.dim()
    flush(rawdata, path)
    rawdata.putInt(ndimension)

    var i = 0
    while (i < ndimension) {
      flush(rawdata, path)
      rawdata.putLong(source.size(i + 1))
      i += 1
    }

    i = 0
    while (i < ndimension) {
      flush(rawdata, path)
      rawdata.putLong(source.stride(i + 1))
      i += 1
    }
    flush(rawdata, path)
    rawdata.putLong(source.storageOffset())

    if (ndimension == 0) {
      writeObject(source, rawdata, path, TYPE_NIL)
    }
    else {
      writeObject(source, rawdata, path, TYPE_FLOAT_STORAGE)
    }

    byteWrite(rawdata, path)
  }

  private def writeSpatialConvolution(source: SpatialConvolution[Double], rawdata: ByteBuffer,
    path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val nInputPlane = source.nInputPlane
    val nOutputPlane = source.nOutputPlane
    val kW = source.kW
    val kH = source.kH
    val dW = source.dW
    val dH = source.dH
    val padW = source.padW
    val padH = source.padH
    val gradBias = source.gradBias
    val fGradInput = source.fGradInput
//    val fInput = source.fInput
    val bias = source.bias
    val weight = source.weight
    val gradWeight = source.gradWeight
    val output = source.output
    val gradInput = source.gradInput
    table.put("gradInput", gradInput)
    table.put("nInputPlane", nInputPlane)
    table.put("nOutputPlane", nOutputPlane)
    table.put("kW", kW)
    table.put("kH", kH)
    table.put("dW", dW)
    table.put("dH", dH)
    table.put("padW", padW)
    table.put("padH", padH)
    table.put("fGradInput", fGradInput)
//    table.put("fInput", fInput)
    table.put("gradBias", gradBias)
    table.put("output", output)
    table.put("bias", bias)
    table.put("weight", weight)
    table.put("gradWeight", gradWeight)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeSpatialMaxPooling(source: SpatialMaxPooling[Double], rawdata: ByteBuffer,
    path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val indices = source.indices
    val ceilMode = source.ceil_mode
    val kW = source.kW
    val kH = source.kH
    val dW = source.dW
    val dH = source.dH
    val padW = source.padW
    val padH = source.padH
    val output = source.output
    val gradInput = source.gradInput
    table.put("gradInput", gradInput)
    table.put("kW", kW)
    table.put("kH", kH)
    table.put("dW", dW)
    table.put("dH", dH)
    table.put("padW", padW)
    table.put("padH", padH)
    table.put("indices", indices)
    table.put("ceil_mode", ceilMode)
    table.put("output", output)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeThreshold(source: Threshold[Double], rawdata: ByteBuffer, path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val value = source.value
    val output = source.output
    val inPlace = source.inPlace
    val gradInput = source.gradInput
    val threshold = source.threshold
    table.put("gradInput", gradInput)
    table.put("val", value)
    table.put("inplace", inPlace)
    table.put("threshold", threshold)
    table.put("output", output)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeConcat(source: Concat[Double], rawdata: ByteBuffer, path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val dimension = source.dimension
    val size = source.getSize()
    val output = source.output
    val train = source.training()
    val gradInput = source.gradInput
    val modules: Map[Double, Module[Double]] = new HashMap()

    for (i <- 1 to source.modules.length) {
      modules.put(i, source.modules(i - 1))
    }

    table.put("gradInput", gradInput)
    table.put("size", size)
    table.put("dimension", dimension)
    table.put("modules", modules.asInstanceOf[Map[Any, Any]])
    table.put("output", output)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeSequential(source: Sequential[Double],
    rawdata: ByteBuffer, path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val output = source.output
    val gradInput = source.gradInput
    val modules: Map[Double, Module[Double]] = new HashMap()

    for (i <- 1 to source.modules.length) {
      modules.put(i, source.modules(i - 1))
    }

    table.put("gradInput", gradInput)
    table.put("modules", modules.asInstanceOf[Map[Any, Any]])
    table.put("output", output)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeDropout(source: Dropout[Double], rawdata: ByteBuffer, path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val p = source.getP()
    val output = source.output
    val noise = source.noise
    val gradInput = source.gradInput
    val train = source.isTraining()

    table.put("gradInput", gradInput)
    table.put("train", train)
    table.put("noise", noise)
    table.put("gradInput", gradInput)
    table.put("output", output)
    table.put("p", p)

    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeView(source: View[Double], rawdata: ByteBuffer, path: Path): Unit = {
    var table: Map[String, Any] = new HashMap()
    val size = source.getSize()
    val output = source.output
    val numElements = source.numElements

    table.put("numElements", numElements)
    table.put("output", output)
    table.put("size", size)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }


  private def writeLinear(source: Linear[Double], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Map[String, Any] = new HashMap()
    val gradBias = source.gradBias
    val output = source.output
    val gradInput = source.gradInput
    val bias = source.bias
    val weight = source.weight
    val gradWeight = source.gradWeight
    table.put("gradBias", gradBias)
    table.put("output", output)
    table.put("gradInput", gradInput)
    table.put("bias", bias)
    table.put("weight", weight)
    table.put("gradWeight", gradWeight)
    writeObject(table.asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }


  private def writeTable(source: Map[Any, Any], rawdata: ByteBuffer, path: Path): Unit = {
    val size = source.size()
    flush(rawdata, path)
    rawdata.putInt(size)

    val it = source.keySet().iterator();
    while (it.hasNext()) {
      var key = it.next();
      if (key.isInstanceOf[String]) {
        writeObject(key.asInstanceOf[String], rawdata, path, TYPE_STRING)
      }
      else if (key.isInstanceOf[Double]) {
        writeObject(key.asInstanceOf[Double], rawdata, path, TYPE_NUMBER)
      }

      if (source.get(key) == null) {
        writeObject(source.get(key), rawdata, path, TYPE_NIL)
      }
      else if (source.get(key).isInstanceOf[Tensor[_]]) {
        writeObject(source.get(key).asInstanceOf[Tensor[Double]], rawdata, path, TYPE_DOUBLE_TENSOR)
      }
      else if (source.get(key).isInstanceOf[Int]) {
        writeObject(source.get(key).asInstanceOf[Int].toDouble, rawdata, path, TYPE_NUMBER)
      }
      else if (source.get(key).isInstanceOf[Double]) {
        writeObject(source.get(key).asInstanceOf[Double], rawdata, path, TYPE_NUMBER)
      }
      else if (source.get(key).isInstanceOf[Boolean]) {
        writeObject(source.get(key).asInstanceOf[Boolean], rawdata, path, TYPE_BOOLEAN)
      }
      else if (source.get(key).isInstanceOf[Map[_, _]]) {
        writeObject(source.get(key).asInstanceOf[Map[Any, Any]], rawdata, path, TYPE_TABLE)
      }
      else if (source.get(key).isInstanceOf[Linear[_]]) {
        writeObject(source.get(key).asInstanceOf[Linear[Double]], rawdata, path, TYPE_LINEAR)
      }
      else if (source.get(key).isInstanceOf[Array[Int]]) {
        writeObject(source.get(key).asInstanceOf[Array[Int]], rawdata, path, TYPE_LONG_STORAGE)
      }

    }
    byteWrite(rawdata, path)
  }

  private def writeFloatStorage(source: Tensor[Float], rawdata: ByteBuffer, path: Path): Unit = {
    val storageLength = source.storage().length()
    flush(rawdata, path)
    rawdata.putLong(storageLength)
    var i = 0
    while (i < storageLength) {
      flush(rawdata, path)
      rawdata.putFloat(source.storage().asInstanceOf[Storage[Float]](i))
      i += 1
    }
    byteWrite(rawdata, path)

  }

  private def writeDoubleStorage(source: Tensor[Double], rawdata: ByteBuffer, path: Path): Unit = {
    val storageLength = source.storage().length()
    flush(rawdata, path)
    rawdata.putLong(storageLength)
    var i = 0
    while (i < storageLength) {
      flush(rawdata, path)
      rawdata.putDouble(source.storage().asInstanceOf[Storage[Double]](i))
      i += 1
    }
    byteWrite(rawdata, path)
  }

  private def writeLongStorage(source: Array[Int], rawdata: ByteBuffer, path: Path): Unit = {
    val storageLength = source.length
    flush(rawdata, path)
    rawdata.putLong(storageLength)
    var i = 0
    while (i < storageLength) {
      flush(rawdata, path)
      rawdata.putLong(source(i))
      i += 1
    }
    byteWrite(rawdata, path)
  }

  // Basic objects
  private def readDoubleStorage(rawData: ByteBuffer): Storage[Double] = {
    val storageLength = rawData.getLong.toInt
    val data = new Array[Double](storageLength)
    var i = 0
    while (i < storageLength) {
      data(i) = rawData.getDouble
      i += 1
    }
    Storage(data)
  }

  private def readFloatStorage(rawData: ByteBuffer): Storage[Float] = {
    val storageLength = rawData.getLong.toInt
    val data = new Array[Float](storageLength)
    var i = 0
    while (i < storageLength) {
      data(i) = rawData.getFloat
      i += 1
    }
    Storage(data)
  }

  private def readLongStorage(rawData: ByteBuffer): Array[Int] = {
    val storageLength = rawData.getLong.toInt
    val data = new Array[Int](storageLength)
    var i = 0
    while (i < storageLength) {
      data(i) = rawData.getLong.toInt
      i += 1
    }
    data
  }

  private def readVersionAndClass(rawData: ByteBuffer): (Int, String) = {
    val version = readString(rawData: ByteBuffer)
    val pattern = "^V (.*)$".r
    version match {
      case pattern(v) => (v.toInt, readString(rawData))
      case _ => (0, version)
    }
  }

  private def readString(rawData: ByteBuffer): String = {
    val length = rawData.getInt()
    val string = new Array[Char](length)
    var i = 0
    while (i < string.length) {
      string(i) = rawData.get.toChar
      i += 1
    }
    new String(string)
  }

  private def readNumber(rawData: ByteBuffer): Double = {
    rawData.getDouble()
  }

  private def readBoolean(rawData: ByteBuffer): Boolean = {
    rawData.getInt == 1
  }

  // Table
  private def readTable(rawData: ByteBuffer, objects: Map[Int, Any]): Map[Any, Any] = {
    val size = rawData.getInt
    val result = new HashMap[Any, Any]()
    var i = 0
    while (i < size) {
      result.put(readObject(rawData, objects), readObject(rawData, objects))
      i += 1
    }
    result
  }

  // Tensor
  private def readDoubleTensor(rawData: ByteBuffer, objects: Map[Int, Any]): Tensor[Double] = {
    val nDimension = rawData.getInt()
    val sizes = new Array[Int](nDimension)
    val strides = new Array[Int](nDimension)
    var i = 0
    while (i < nDimension) {
      sizes(i) = rawData.getLong.toInt
      i += 1
    }
    i = 0
    while (i < nDimension) {
      strides(i) = rawData.getLong.toInt
      i += 1
    }

    val offset = rawData.getLong.toInt


    val storage = readObject(rawData, objects).asInstanceOf[Storage[Double]]
    Tensor(storage, offset, sizes, strides)
  }

  // Tensor float
  private def readFloatTensor(rawData: ByteBuffer, objects: Map[Int, Any]): Tensor[Float] = {
    val nDimension = rawData.getInt()
    val sizes = new Array[Int](nDimension)
    val strides = new Array[Int](nDimension)
    var i = 0
    while (i < nDimension) {
      sizes(i) = rawData.getLong.toInt
      i += 1
    }
    i = 0
    while (i < nDimension) {
      strides(i) = rawData.getLong.toInt
      i += 1
    }

    val offset = rawData.getLong.toInt


    val storage = readObject(rawData, objects).asInstanceOf[Storage[Float]]
    Tensor(storage, offset, sizes, strides)
  }

  // Modules
  private def readSpatialMaxPooling(
    rawData: ByteBuffer, objects: Map[Int, Any]): SpatialMaxPooling[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val padW = elements.get("padW").asInstanceOf[Double].toInt
    val padH = elements.get("padH").asInstanceOf[Double].toInt
    val indices = elements.get("indices").asInstanceOf[Tensor[Double]]
    val dW = elements.get("dW").asInstanceOf[Double].toInt
    val dH = elements.get("dH").asInstanceOf[Double].toInt
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val ceilMode = elements.get("ceil_mode").asInstanceOf[Boolean]
    val kW = elements.get("kW").asInstanceOf[Double].toInt
    val kH = elements.get("kH").asInstanceOf[Double].toInt
    val result = new SpatialMaxPooling[Double](kW, kH, dW, dH, padW, padH)
    result.ceil_mode = ceilMode
    result.output.resizeAs(output)
    result.output.copy(output)
    result.indices.resizeAs(indices)
    result.indices.copy(indices)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result
  }

  private def readSpatialAveragePooling(
    rawData: ByteBuffer, objects: Map[Int, Any]): SpatialAveragePooling[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val padW = elements.get("padW").asInstanceOf[Double].toInt
    val padH = elements.get("padH").asInstanceOf[Double].toInt
    val dW = elements.get("dW").asInstanceOf[Double].toInt
    val dH = elements.get("dH").asInstanceOf[Double].toInt
    val ceilMode = elements.get("ceil_mode").asInstanceOf[Boolean]
    val kW = elements.get("kW").asInstanceOf[Double].toInt
    val kH = elements.get("kH").asInstanceOf[Double].toInt
    val countIncludePad = elements.get("count_include_pad").asInstanceOf[Boolean]
    val divide = elements.get("divide").asInstanceOf[Boolean]
    val result = new SpatialAveragePooling[Double](kW, kH, dW, dH, padW, padH, ceilMode,
      countIncludePad, divide)
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result
  }

  private def readConcat(rawData: ByteBuffer, objects: Map[Int, Any]): Concat[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    // size array will be adjust to the input in the training
    val size = elements.get("size").asInstanceOf[Array[Int]]
    val dimension = elements.get("dimension").asInstanceOf[Double].toInt
    val train = elements.get("train").asInstanceOf[Boolean] // what's this?
    val modules = elements.get("modules").asInstanceOf[Map[Any, Any]]
    val result = new Concat[Double](dimension)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.output.resizeAs(output)
    result.output.copy(output)

    for (m <- readModules(modules)) {
      result.modules += m
    }
    result
  }

  private def readDropout(rawData: ByteBuffer, objects: Map[Int, Any]): Dropout[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val p = elements.get("p").asInstanceOf[Double]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val noise = elements.get("noise").asInstanceOf[Tensor[Double]]
    val train = elements.get("train").asInstanceOf[Boolean]

    val result = new Dropout[Double](p, false, true)
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.noise.resizeAs(noise)
    result.noise.copy(noise)
    if (train) result.training() else result.evaluate()

    result
  }

  private def readLinear(rawData: ByteBuffer, objects: Map[Int, Any]): Linear[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradBias = elements.get("gradBias").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val bias = elements.get("bias").asInstanceOf[Tensor[Double]]
    val weight = elements.get("weight").asInstanceOf[Tensor[Double]]
    val gradWeight = elements.get("gradWeight").asInstanceOf[Tensor[Double]]
    val result = new Linear[Double](weight.size(2), weight.size(1))
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradBias.resizeAs(gradBias)
    result.gradBias.copy(gradBias)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.bias.resizeAs(bias)
    result.bias.copy(bias)
    result.weight.resizeAs(weight)
    result.weight.copy(weight)
    result.gradWeight.resizeAs(gradWeight)
    result.gradWeight.copy(gradWeight)
    result
  }

  private def readSpatialConvolutionMap(
    rawData: ByteBuffer, objects: Map[Int, Any]): SpatialConvolutionMap[Double] = {

    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val padH = elements.get("padH").asInstanceOf[Double].toInt
    val padW = elements.get("padW").asInstanceOf[Double].toInt
    val dH = elements.get("dH").asInstanceOf[Double].toInt
    val dW = elements.get("dW").asInstanceOf[Double].toInt
    val kH = elements.get("kH").asInstanceOf[Double].toInt
    val kW = elements.get("kW").asInstanceOf[Double].toInt
    val connTable = elements.get("connTable").asInstanceOf[Tensor[Double]]
    val gradBias = elements.get("gradBias").asInstanceOf[Tensor[Double]]
    val weight = elements.get("weight").asInstanceOf[Tensor[Double]]
    //    val finput = elements.get("finput").asInstanceOf[Tensor[Double]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val bias = elements.get("bias").asInstanceOf[Tensor[Double]]
    val gradWeight = elements.get("gradWeight").asInstanceOf[Tensor[Double]]
    //    val fgradInput = elements.get("fgradInput").asInstanceOf[Tensor[Double]]
    val result = new SpatialConvolutionMap[Double](connTable, kW, kH, dW, dH, padW, padH)
    result.gradBias.resizeAs(gradBias)
    result.gradBias.copy(gradBias)
    result.weight.resizeAs(weight)
    result.weight.copy(weight)
    //    result.fInput.resizeAs(finput)
    //    result.fInput.copy(finput)
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.bias.resizeAs(bias)
    result.bias.copy(bias)
    result.gradWeight.resizeAs(gradWeight)
    result.gradWeight.copy(gradWeight)
    //    result.fGradInput.resizeAs(fgradInput)
    //    result.fGradInput.copy(fgradInput)
    result
  }

  private def readBatchNormalization(
    rawData: ByteBuffer, objects: Map[Int, Any]): BatchNormalization[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val eps = elements.get("eps").asInstanceOf[Double]
    val momentum = elements.get("momentum").asInstanceOf[Double]
    val affine = elements.get("affine").asInstanceOf[Boolean]
    val gradBias = elements.get("gradBias").asInstanceOf[Tensor[Double]]
    val weight = elements.get("weight").asInstanceOf[Tensor[Double]]
    val runningMean = elements.get("running_mean").asInstanceOf[Tensor[Double]]
    val runningVar = elements.get("running_var").asInstanceOf[Tensor[Double]]
    val saveMean = elements.get("save_mean").asInstanceOf[Tensor[Double]]
    val saveStd = elements.get("save_std").asInstanceOf[Tensor[Double]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val bias = elements.get("bias").asInstanceOf[Tensor[Double]]
    val gradWeight = elements.get("gradWeight").asInstanceOf[Tensor[Double]]
    val nOutput = runningMean.size(1)
    val result = new BatchNormalization[Double](nOutput, eps, momentum, affine)
    result.gradBias.resizeAs(gradBias)
    result.gradBias.copy(gradBias)
    result.weight.resizeAs(weight)
    result.weight.copy(weight)
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.bias.resizeAs(bias)
    result.bias.copy(bias)
    result.gradWeight.resizeAs(gradWeight)
    result.gradWeight.copy(gradWeight)
    result.runningMean.resizeAs(runningMean)
    result.runningMean.copy(runningMean)
    result.runningVar.resizeAs(runningVar)
    result.runningVar.copy(runningVar)
    result.saveMean.resizeAs(saveMean)
    result.saveMean.copy(saveMean)
    result.saveStd.resizeAs(saveStd)
    result.saveStd.copy(saveStd)

    result
  }

  private def readSpatialBatchNormalization(
    rawData: ByteBuffer, objects: Map[Int, Any]): SpatialBatchNormalization[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val eps = elements.get("eps").asInstanceOf[Double]
    val momentum = elements.get("momentum").asInstanceOf[Double]
    val affine = elements.get("affine").asInstanceOf[Boolean]
    val gradBias = elements.get("gradBias").asInstanceOf[Tensor[Double]]
    val weight = elements.get("weight").asInstanceOf[Tensor[Double]]
    val runningMean = elements.get("running_mean").asInstanceOf[Tensor[Double]]
    val runningVar = elements.get("running_var").asInstanceOf[Tensor[Double]]
    val saveMean = elements.get("save_mean").asInstanceOf[Tensor[Double]]
    val saveStd = elements.get("save_std").asInstanceOf[Tensor[Double]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val bias = elements.get("bias").asInstanceOf[Tensor[Double]]
    val gradWeight = elements.get("gradWeight").asInstanceOf[Tensor[Double]]
    val nOutput = runningMean.size(1)
    val result = new SpatialBatchNormalization[Double](nOutput, eps, momentum, affine)
    result.gradBias.resizeAs(gradBias)
    result.gradBias.copy(gradBias)
    result.weight.resizeAs(weight)
    result.weight.copy(weight)
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.bias.resizeAs(bias)
    result.bias.copy(bias)
    result.gradWeight.resizeAs(gradWeight)
    result.gradWeight.copy(gradWeight)
    result.runningMean.resizeAs(runningMean)
    result.runningMean.copy(runningMean)
    result.runningVar.resizeAs(runningVar)
    result.runningVar.copy(runningVar)
    if (null != saveMean) {
      result.saveMean.resizeAs(saveMean)
      result.saveMean.copy(saveMean)
    }
    if (null != saveStd) {
      result.saveStd.resizeAs(saveStd)
      result.saveStd.copy(saveStd)
    }

    result
  }

  private def readThreshold(rawData: ByteBuffer, objects: Map[Int, Any]): Threshold[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val result = new Threshold[Double]
    val value = elements.get("val").asInstanceOf[Double]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val inPlace = elements.get("inplace").asInstanceOf[Boolean]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val threshold = elements.get("threshold").asInstanceOf[Double]
    result.value = value
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.inPlace = inPlace
    result.threshold = threshold
    result
  }

  private def readLogSoftMax(rawData: ByteBuffer, objects: Map[Int, Any]): LogSoftMax[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val result = new LogSoftMax[Double]
    result.output = elements.get("output").asInstanceOf[Tensor[Double]]
    result.gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    result
  }

  private def readView(rawData: ByteBuffer, objects: Map[Int, Any]): View[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val size = elements.get("size").asInstanceOf[Array[Int]]
    val result = new View[Double](size)
    if (elements.containsKey("output")) {
      val output = elements.get("output").asInstanceOf[Tensor[Double]]
      result.output.resizeAs(output)
      result.output.copy(output)
    }
    val numElements = elements.get("numElements").asInstanceOf[Double].toInt
    val numInputDims = elements.get("numInputDims").asInstanceOf[Double].toInt
    result.setNumInputDims(numInputDims)
    require(result.numElements == numElements, "Invalid view file")
    result
  }

  private def readSpatialZeroPadding(
    rawData: ByteBuffer, objects: Map[Int, Any]): SpatialZeroPadding[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val result = new SpatialZeroPadding[Double](
      elements.get("pad_l").asInstanceOf[Double].toInt,
      elements.get("pad_r").asInstanceOf[Double].toInt,
      elements.get("pad_t").asInstanceOf[Double].toInt,
      elements.get("pad_b").asInstanceOf[Double].toInt
    )
    result.output = elements.get("output").asInstanceOf[Tensor[Double]]
    result.gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    result
  }

  private def readReLU(rawData: ByteBuffer, objects: Map[Int, Any]): ReLU[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val result = new ReLU[Double]
    result.value = elements.get("val").asInstanceOf[Double]
    result.output = elements.get("output").asInstanceOf[Tensor[Double]]
    result.inPlace = elements.get("inplace").asInstanceOf[Boolean]
    result.gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    result.threshold = elements.get("threshold").asInstanceOf[Double]
    result
  }

  private def readTanh(rawData: ByteBuffer, objects: Map[Int, Any]): Tanh[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val result = new Tanh[Double]
    result
  }

  private def readReshape(rawData: ByteBuffer, objects: Map[Int, Any]): Reshape[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val size = elements.get("size").asInstanceOf[Array[Int]]
    val result = new Reshape[Double](size)
    result
  }

  private def readSpatialConvolution(
    rawData: ByteBuffer, objects: Map[Int, Any]): SpatialConvolution[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[String, Any]]
    val padH = elements.get("padH").asInstanceOf[Double].toInt
    val padW = elements.get("padW").asInstanceOf[Double].toInt
    val dH = elements.get("dH").asInstanceOf[Double].toInt
    val dW = elements.get("dW").asInstanceOf[Double].toInt
    val kH = elements.get("kH").asInstanceOf[Double].toInt
    val kW = elements.get("kW").asInstanceOf[Double].toInt
    val nInputPlane = elements.get("nInputPlane").asInstanceOf[Double].toInt
    val nOutputPlane = elements.get("nOutputPlane").asInstanceOf[Double].toInt
    val gradBias = elements.get("gradBias").asInstanceOf[Tensor[Double]]
    val weight = elements.get("weight").asInstanceOf[Tensor[Double]]
    val finput = elements.get("finput").asInstanceOf[Tensor[Double]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
    val bias = elements.get("bias").asInstanceOf[Tensor[Double]]
    val gradWeight = elements.get("gradWeight").asInstanceOf[Tensor[Double]]
    val fgradInput = elements.get("fgradInput").asInstanceOf[Tensor[Double]]
    val result = new SpatialConvolution[Double](
      nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    result.gradBias.resizeAs(gradBias)
    result.gradBias.copy(gradBias)
    result.weight.resizeAs(weight)
    result.weight.copy(weight)
//    if (finput != null) {
//      result.fInput.resizeAs(finput)
//      result.fInput.copy(finput)
//    }
    result.output.resizeAs(output)
    result.output.copy(output)
    result.gradInput.resizeAs(gradInput)
    result.gradInput.copy(gradInput)
    result.bias.resizeAs(bias)
    result.bias.copy(bias)
    result.gradWeight.resizeAs(gradWeight)
    result.gradWeight.copy(gradWeight)
    if (fgradInput != null) {
      result.fGradInput.resizeAs(fgradInput)
      result.fGradInput.copy(fgradInput)
    }
    result
  }

  private def readSequentialModule(
    rawData: ByteBuffer, objects: Map[Int, Any]): Sequential[Double] = {
    val elements = readObject(rawData, objects).asInstanceOf[Map[Any, Any]]
    val output = elements.get("output").asInstanceOf[Tensor[Double]]
    val modules = elements.get("modules").asInstanceOf[Map[Any, Any]]
    val result = new Sequential[Double]()
    if (null != output) {
      result.output.resizeAs(output)
      result.output.copy(output)
    }
    if (elements.containsKey("gradInput")) {
      val gradInput = elements.get("gradInput").asInstanceOf[Tensor[Double]]
      if (null != gradInput) {
        result.gradInput.resizeAs(gradInput)
        result.gradInput.copy(gradInput)
      }
    }

    for (m <- readModules(modules)) {
      result.modules += m
    }
    result
  }

  private def readModules(modules: Map[Any, Any]): Array[Module[Double]] = {
    val moduleLength = modules.keySet().size()
    val modulesArray = new Array[Module[Double]](moduleLength)
    for (k <- modules.keySet().toArray) {
      val key = k.asInstanceOf[Double]
      modulesArray(key.toInt - 1) = modules.get(key).asInstanceOf[Module[Double]]
    }
    modulesArray
  }
}
