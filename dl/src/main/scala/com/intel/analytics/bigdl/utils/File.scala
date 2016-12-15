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

package com.intel.analytics.bigdl.utils

import java.io._
import java.nio._
import java.nio.file._

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.mutable


sealed abstract class TorchObject(val typeId: Int)

object TorchObject {

  case object TYPE_NIL extends TorchObject(0)

  case object TYPE_NUMBER extends TorchObject(1)

  case object TYPE_FLOAT_TENSOR extends TorchObject(4)

  case object TYPE_DOUBLE_TENSOR extends TorchObject(4)

  case object TYPE_DOUBLE_STORAGE extends TorchObject(4)

  case object TYPE_FLOAT_STORAGE extends TorchObject(4)

  case object TYPE_LONG_STORAGE extends TorchObject(4)

  case object TYPE_MODULE extends TorchObject(4)

  case object TYPE_STRING extends TorchObject(2)

  case object TYPE_BOOLEAN extends TorchObject(5)

  case object TYPE_TABLE extends TorchObject(3)
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
  def loadTorch[T](fileName: String): T = {
    val path = Paths.get(fileName)
    val rawData = ByteBuffer.wrap(Files.readAllBytes(path))
    rawData.order(ByteOrder.LITTLE_ENDIAN)
    val objects: mutable.Map[Int, Any] = new mutable.HashMap()
    readObject(rawData, objects).asInstanceOf[T]
  }

  /**
   * Save torch object into a torch format file
   *
   * @param source
   * @param fileName
   * @param objectType
   */
  def saveTorch(
      source: Any,
      fileName: String,
      objectType: TorchObject,
      overWrite: Boolean = false): Unit = {
    val file = new File(fileName)
    if (file.exists()) {
      require(file.isFile(), s"$fileName is not a file")
      if (!overWrite) {
        throw new FileAlreadyExistsException(fileName)
      } else { // clear the file
        val fw = new FileWriter(file)
        fw.write("")
        fw.close()
      }
    } else {
      file.createNewFile()
    }
    val capacity = 300000
    val buffer = ByteBuffer.allocate(capacity)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    writeObject(source: Any, buffer, file.toPath, objectType)
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
   * Load object from a Java object file
   *
   * @param fileName
   * @tparam T
   */
  def load[T](fileName: String): T = {
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

  private def createInstanceFor(className: String, args: mutable.Seq[(Class[_], AnyRef)]): Any = {
    val types = args.map(_._1).toArray
    val values = args.map(_._2).toArray
    val constructor = Class.forName(className).getDeclaredConstructor(types: _*)
    constructor.setAccessible(true)
    val obj = constructor.newInstance(values: _*)
    obj
  }

  private def readModuleWithType[T: ClassTag](
      moduleName: String,
      elements: Table)(implicit ev: TensorNumeric[T]): Any = {
    val module = moduleName match {
      case "nn.BatchNormalization" => readBatchNormalizationWithType(elements)
      case "nn.CAddTable" => CAddTable(elements.getOrElse("inplace", false))
      case "nn.Concat" => readConcatWithType(elements)
      case "nn.ConcatTable" => readConcatTableWithType(elements)
      case "nn.Dropout" => readDropoutWithType(elements)
      case "nn.Linear" => readLinearWithType(elements)
      case "nn.ReLU" => ReLU(elements("inplace").asInstanceOf[Boolean])
      case "nn.Reshape" => Reshape(elements("size").asInstanceOf[Array[Int]])
      case "nn.Sequential" => readSequentialModuleWithType[T](elements)
      case "nn.SpatialMaxPooling" => readSpatialMaxPoolingWithType(elements)
      case "nn.SpatialAveragePooling" => readSpatialAveragePoolingWithType(elements)
      case "nn.SpatialBatchNormalization" => readSpatialBatchNormalizationWithType(elements)
      case "nn.SpatialConvolution" => readSpatialConvolutionWithType(elements)
      case "nn.SpatialConvolutionMap" => readSpatialConvolutionMapWithType(elements)
      case "nn.SpatialConvolutionMM" => readSpatialConvolutionWithType(elements)
      case "nn.SpatialZeroPadding" => readSpatialZeroPaddingWithType(elements)
      case "nn.Threshold" => readThresholdWithType(elements)
      case "nn.View" => readViewWithType(elements)
      case _ => try {
        val classTag =
          if (elements("_type") == "torch.FloatTensor") ClassTag.Float else ClassTag.Double
        val bigDlName = "com.intel.analytics.bigdl." + moduleName
        // Use reflection to load all parameter-free module
        val args: Array[(Class[_], AnyRef)] = Array((classOf[ClassTag[_]], classTag),
          (classOf[TensorNumeric[_]], ev))
        createInstanceFor(bigDlName, args)
      } catch {
        case _ => throw new IllegalArgumentException(s"unsupported module $moduleName")
      }
    }
    module
  }

  private def readModule(
    moduleName: String,
    rawData: ByteBuffer,
    objects: mutable.Map[Int, Any]): Any = {
    val elements = readObject(rawData, objects).asInstanceOf[Table]
    val moduleType = elements.getOrElse("_type", "")
    if(moduleType == "torch.FloatTensor") {
      readModuleWithType[Float](moduleName, elements)
    } else if (moduleType == "torch.DoubleTensor") {
      readModuleWithType[Double](moduleName, elements)
    } else {
      throw new Error(s"unkown module type $moduleType")
    }

  }

  private def readObject(rawData: ByteBuffer, objects: mutable.Map[Int, Any]): Any = {
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
        if (objects.contains(indexId)) {
          objects(indexId)
        } else {
          val (versionNumber, className) = readVersionAndClass(rawData)
          // Todo: Use reflection to do this is better
          val result = className match {
            case "torch.FloatTensor" => readFloatTensor(rawData, objects)
            case "torch.DoubleTensor" => readDoubleTensor(rawData, objects)
            case "torch.LongTensor" => readLongTensor(rawData, objects)
            case "torch.DoubleStorage" => readDoubleStorage(rawData)
            case "torch.FloatStorage" => readFloatStorage(rawData)
            case "torch.LongStorage" => readLongStorage(rawData)
            case _ => readModule(className, rawData, objects)
          }
          objects.put(indexId, result)
          result
        }
      case TYPE_TABLE =>
        val indexId = rawData.getInt()
        if (objects.contains(indexId)) {
          objects(indexId)
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

  private def writeModule(
      module: AbstractModule[_, _, _],
      rawData: ByteBuffer, path: Path): Unit = {
    module match {
      case m: Linear[_] =>
        writeVersionAndClass("V 1", "nn.Linear", rawData, path)
        writeLinear(m, rawData, path)
      case m: SpatialConvolution[_] =>
        writeVersionAndClass("V 1", "nn.SpatialConvolutionMM", rawData, path)
        writeSpatialConvolution(m, rawData, path)
      case m: SpatialMaxPooling[_] =>
        writeVersionAndClass("V 1", "nn.SpatialMaxPooling", rawData, path)
        writeSpatialMaxPooling(m, rawData, path)
      case m: ReLU[_] =>
        writeVersionAndClass("V 1", "nn.ReLU", rawData, path)
        writeReLU(m, rawData, path)
      case m: Threshold[_] =>
        writeVersionAndClass("V 1", "nn.Threshold", rawData, path)
        writeThreshold(m, rawData, path)
      case m: Concat[_] =>
        writeVersionAndClass("V 1", "nn.Concat", rawData, path)
        writeConcat(m, rawData, path)
      case m: Sequential[_] =>
        writeVersionAndClass("V 1", "nn.Sequential", rawData, path)
        writeSequential(m, rawData, path)
      case m: Dropout[_] =>
        writeVersionAndClass("V 1", "nn.Dropout", rawData, path)
        writeDropout(m, rawData, path)
      case m: View[_] =>
        writeVersionAndClass("V 1", "nn.View", rawData, path)
        writeView(m, rawData, path)
      case m: LogSoftMax[_] =>
        writeVersionAndClass("V 1", "nn.LogSoftMax", rawData, path)
        writeLogSoftMax(m, rawData, path)
      case _ => throw new Error(s"Unimplemented module $module")
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
      case TYPE_MODULE =>
        i = i + 1
        rawdata.putInt(i)
        writeModule(source.asInstanceOf[AbstractModule[_, _, _]], rawdata, path)
      case TYPE_TABLE =>
        i = i + 1
        rawdata.putInt(i)
        source match {
          case s: mutable.Map[Any, Any] =>
            writeTable(s, rawdata, path)
          case s: Table =>
            writeTable(source.asInstanceOf[Table].getState(), rawdata, path)
          case _ => throw new Error(s"Unknown table $source")
        }
      case _ => throw new IllegalArgumentException(objectType.toString)

    }
    byteWrite(rawdata, path)
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
    val tmp = if (source) 1 else 0
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

  private def writeGeneralParameters(
      source: AbstractModule[_, _, _],
      table: Table): Table = {
    table("gradInput") = source.gradInput
    table("output") = source.output
    table("_type") = source.getNumericType() match {
      case DoubleType => "torch.DoubleTensor"
      case FloatType => "torch.FloatTensor"
      case _ =>
        throw new IllegalArgumentException(s"Unknown type ${source.getNumericType()}")
    }
    table
  }

  private def writeSpatialConvolution(source: SpatialConvolution[_], rawdata: ByteBuffer,
    path: Path): Unit = {
    require(source.nGroup == 1, "nGroup is not supported in torch")
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("nInputPlane") = source.nInputPlane
    table("nOutputPlane") = source.nOutputPlane
    table("kW") = source.kernelW
    table("kH") = source.kernelH
    table("dW") = source.strideW
    table("dH") = source.strideH
    table("padW") = source.padW
    table("padH") = source.padH
    table("fGradInput") = source.fGradInput
    table("fInput") = source.fInput
    table("gradBias") = source.gradBias
    table("bias") = source.bias
    table("weight") = source.weight.clone().resize(source.nOutputPlane,
      source.nInputPlane * source.kernelH * source.kernelW)
    table("gradWeight") = source.gradWeight.clone().resize(source.nOutputPlane,
      source.nInputPlane * source.kernelH * source.kernelW)
    if (!source.propagateBack) table("gradInput") = null
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeSpatialMaxPooling(source: SpatialMaxPooling[_], rawdata: ByteBuffer,
    path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("kW") = source.kW
    table("kH") = source.kH
    table("dW") = source.dW
    table("dH") = source.dH
    table("padW") = source.padW
    table("padH") = source.padH
    table("indices") = source.indices
    table("ceil_mode") = source.ceil_mode
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeThreshold(source: Threshold[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("val") = source.value
    table("inplace") = source.inPlace
    table("threshold") = source.threshold
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeConcat(source: Concat[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    val modules = T()
    for (i <- 1 to source.modules.length) {
      modules(i) = source.modules(i - 1)
    }
    table("size") = source.getSize()
    table("dimension") = source.dimension
    table("modules") = modules
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeSequential(source: Sequential[_],
    rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    val modules = T()
    for (i <- 1 to source.modules.length) {
      modules(i) = source.modules(i - 1)
    }
    table("modules") = modules
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeDropout(source: Dropout[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("noise") = source.noise
    table("p") = source.getP()
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeView(source: View[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("numElements") = source.numElements
    table("size") = source.getSize()
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeReLU(source: ReLU[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("val") = source.value
    table("threshold") = source.threshold
    table("inplace") = source.inPlace
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeLogSoftMax(source: LogSoftMax[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def writeLinear(source: Linear[_], rawdata: ByteBuffer, path: Path): Unit = {
    val table: Table = T()
    writeGeneralParameters(source, table)
    table("gradBias") = source.gradBias
    table("bias") = source.bias
    table("weight") = source.weight
    table("gradWeight") = source.gradWeight
    writeObject(table, rawdata, path, TYPE_TABLE)
    byteWrite(rawdata, path)
  }

  private def isDoubleTensor(tensor: Tensor[_]): Boolean = {
    tensor.getType() match {
      case FloatType => false
      case DoubleType => true
      case _ => throw new IllegalArgumentException
    }
  }

  private def writeTable(source: mutable.Map[Any, Any], rawdata: ByteBuffer, path: Path): Unit = {
    val size = source.size
    flush(rawdata, path)
    rawdata.putInt(size)

    val it = source.keySet.toIterator
    while (it.hasNext) {
      val key = it.next()
      key match {
        case k: String =>
          writeObject(k, rawdata, path, TYPE_STRING)
        case d: Double =>
          writeObject(d, rawdata, path, TYPE_NUMBER)
        case f: Float =>
          writeObject(f.toDouble, rawdata, path, TYPE_NUMBER)
        case i: Int =>
          writeObject(i.toDouble, rawdata, path, TYPE_NUMBER)
        case _ => new Error(s"Unsupported key type $key")
      }

      val sourceKey = source.getOrElse(key, null)
      sourceKey match {
        case s: Tensor[_] =>
          if (isDoubleTensor(s)) {
            writeObject(s.asInstanceOf[Tensor[Double]], rawdata, path, TYPE_DOUBLE_TENSOR)
          } else {
            writeObject(s.asInstanceOf[Tensor[Float]], rawdata, path, TYPE_FLOAT_TENSOR)
          }
        case s: Table =>
          writeObject(s.getState(), rawdata, path, TYPE_TABLE)
        case s: Int =>
          writeObject(s.toDouble, rawdata, path, TYPE_NUMBER)
        case s: Float =>
          writeObject(s.toDouble, rawdata, path, TYPE_NUMBER)
        case s: Double =>
          writeObject(s, rawdata, path, TYPE_NUMBER)
        case s: Boolean =>
          writeObject(s, rawdata, path, TYPE_BOOLEAN)
        case s: String =>
          writeObject(s, rawdata, path, TYPE_STRING)
        case s: mutable.Map[_, _] =>
          writeObject(s, rawdata, path, TYPE_TABLE)
        case s: Table =>
          writeObject(s, rawdata, path, TYPE_TABLE)
        case s: Array[Int] =>
          writeObject(s, rawdata, path, TYPE_LONG_STORAGE)
        case s: AbstractModule[_, _, _] =>
          writeObject(s, rawdata, path, TYPE_MODULE)
        case null =>
          writeObject(sourceKey, rawdata, path, TYPE_NIL)
        case _ => new Error(s"Unsupported value type $key")
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
      rawdata.putFloat(source.storage()(i))
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
      rawdata.putDouble(source.storage()(i))
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
  private def readTable(
      rawData: ByteBuffer,
      objects: mutable.Map[Int, Any]): Table = {
    val size = rawData.getInt
    val result = T()
    var i = 0
    while (i < size) {
      val key = readObject(rawData, objects)
      val value = readObject(rawData, objects)
      key match {
        case d: Double if d % 1 == 0 =>
          result(d.toInt) = value
        case _ =>
          result(key) = value
      }
      i += 1
    }
    result
  }

  // Tensor
  private def readDoubleTensor(
      rawData: ByteBuffer,
      objects: mutable.Map[Int, Any]): Tensor[Double] = {
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
  private def readFloatTensor(
      rawData: ByteBuffer,
      objects: mutable.Map[Int, Any]): Tensor[Float] = {
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

  // Tensor long
  private def readLongTensor(
      rawData: ByteBuffer,
      objects: mutable.Map[Int, Any]): Tensor[Double] = {
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


    val longStorage = readObject(rawData, objects).asInstanceOf[Array[Int]]

    val storageData : Array[Double] = new Array[Double](longStorage.length)
    i = 0
    while(i < storageData.length) {
      storageData(i) = longStorage(i)
      i += 1
    }
    val storage = Storage[Double](storageData)
    Tensor(storage, offset, sizes, strides)
  }

  private def readSpatialMaxPoolingWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): SpatialMaxPooling[T] = {
    val result = SpatialMaxPooling[T](
      kW = elements[Double]("kW").toInt,
      kH = elements[Double]("kH").toInt,
      dW = elements[Double]("dW").toInt,
      dH = elements[Double]("dH").toInt,
      padW = elements.getOrElse("padW", 0.0).toInt,
      padH = elements.getOrElse("padH", 0.0).toInt
    )
    val ceilMode = elements("ceil_mode").asInstanceOf[Boolean]
    if (ceilMode) result.ceil() else result.floor()
    result
  }

  private def readSpatialAveragePoolingWithType[T: ClassTag](
      elements: Table)(
      implicit ev: TensorNumeric[T]): SpatialAveragePooling[T] = {
    val result = SpatialAveragePooling[T](
      kW = elements[Double]("kW").toInt,
      kH = elements[Double]("kH").toInt,
      dW = elements.getOrElse("dW", 1.0).toInt,
      dH = elements.getOrElse("dH", 1.0).toInt,
      padW = elements.getOrElse("padW", 0.0).toInt,
      padH = elements.getOrElse("padH", 0.0).toInt,
      ceilMode = elements.getOrElse("ceil_mode", false),
      countIncludePad = elements.getOrElse("count_include_pad", true),
      divide = elements.getOrElse("divide", true)
    )
    result
  }

  private def readConcatWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): Concat[T] = {
    val modules = elements[Table]("modules")
    val result = Concat[T](elements("dimension").asInstanceOf[Double].toInt)

    for (i <- 1 to modules.length()) {
      result.add(modules(i))
    }
    result
  }

  private def readDropoutWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): Dropout[T] = {
    val result = Dropout[T](
      initP = elements.getOrElse("p", 0.5),
      inplace = elements.getOrElse("inplace", false),
      scale = elements.getOrElse("", true)
    )
    result
  }

  private def readLinearWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]) : Linear[T] = {
    val bias = elements("bias").asInstanceOf[Tensor[T]]
    val weight = elements("weight").asInstanceOf[Tensor[T]]
    val result = Linear[T](weight.size(2), weight.size(1))
    result.bias.copy(bias)
    result.weight.copy(weight)
    result

  }

  private def readSpatialConvolutionMapWithType[T: ClassTag](
      elements: Table)(
      implicit ev: TensorNumeric[T]): SpatialConvolutionMap[T] = {
    val weight = elements.getOrElse("weight", null).asInstanceOf[Tensor[T]]
    val bias = elements.getOrElse("bias", null).asInstanceOf[Tensor[T]]
    val result = SpatialConvolutionMap[T](
      connTable = elements("connTable").asInstanceOf[Tensor[T]],
      kW = elements[Double]("kW").toInt,
      kH = elements[Double]("kH").toInt,
      dW = elements.getOrElse("dW", 1.0).toInt,
      dH = elements.getOrElse("dH", 1.0).toInt,
      padW = elements.getOrElse("padW", 0.0).toInt,
      padH = elements.getOrElse("padH", 0.0).toInt
    )
    result.weight.copy(weight)
    result.bias.copy(bias)
    result
  }

  private def readBatchNormalizationWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): BatchNormalization[T] = {
    val weight = elements("weight").asInstanceOf[Tensor[T]]
    val runningMean = elements("running_mean").asInstanceOf[Tensor[T]]
    val runningVar = elements("running_var").asInstanceOf[Tensor[T]]
    val bias = elements("bias").asInstanceOf[Tensor[T]]
    val result = new BatchNormalization[T](
      nOutput = runningMean.size(1),
      eps = elements.getOrElse("eps", 1e-5),
      momentum = elements.getOrElse("momentum", 0.1),
      affine = elements.getOrElse("affine", true)
    )
    result.weight.copy(weight)
    result.bias.copy(bias)
    result.runningMean.copy(runningMean)
    result.runningVar.copy(runningVar)

    result
  }

  private def readSpatialBatchNormalizationWithType[T: ClassTag](
      elements: Table)(
      implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    val weight = elements("weight").asInstanceOf[Tensor[T]]
    val runningMean = elements("running_mean").asInstanceOf[Tensor[T]]
    val runningVar = elements("running_var").asInstanceOf[Tensor[T]]
    val bias = elements("bias").asInstanceOf[Tensor[T]]
    val result = new SpatialBatchNormalization[T](
      nOutput = runningMean.size(1),
      eps = elements.getOrElse("eps", 1e-5),
      momentum = elements.getOrElse("momentum", 0.1),
      affine = elements.getOrElse("affine", true)
    )
    result.weight.copy(weight)
    result.bias.copy(bias)
    result.runningMean.copy(runningMean)
    result.runningVar.copy(runningVar)

    result
  }

  private def readConcatTableWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): ConcatTable[T] = {
    val result = new ConcatTable[T]()
    val modules = elements[Table]("modules")
    for (i <- 1 to modules.length()) {
      result.add(modules(i))
    }
    result
  }

  private def readThresholdWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): Threshold[T] = {
    Threshold[T](
      th = elements.getOrElse("threshold", 1e-6),
      v = elements.getOrElse("val", 0.0),
      ip = elements.getOrElse("inplace", false)
    )
  }

  private def readViewWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): View[T] = {
    val result = View[T](elements("size").asInstanceOf[Array[Int]])
    result.setNumInputDims(elements.getOrElse("numInputDims", 0.0).toInt)
    val numElements = elements.getOrElse("numElements", - 1.0).toInt
    require(result.numElements == numElements, "Invalid view file")
    result
  }

  private def readSpatialZeroPaddingWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): SpatialZeroPadding[T] = {
    val result = SpatialZeroPadding[T](
      elements[Double]("pad_l").toInt,
      elements[Double]("pad_r").toInt,
      elements[Double]("pad_t").toInt,
      elements[Double]("pad_b").toInt
    )
    result
  }

  private def readSpatialConvolutionWithType[T: ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    val propagateBack = if (null == elements("gradInput")) false else true
    val result = SpatialConvolution[T](
      nInputPlane = elements[Double]("nInputPlane").toInt,
      nOutputPlane = elements[Double]("nOutputPlane").toInt,
      kernelW = elements[Double]("kW").toInt,
      kernelH = elements[Double]("kH").toInt,
      strideW = elements.getOrElse("dW", 1.0).toInt,
      strideH = elements.getOrElse("dH", 1.0).toInt,
      padW = elements.getOrElse("padW", 0.0).toInt,
      padH = elements.getOrElse("padH", 0.0).toInt,
      nGroup = 1,
      propagateBack = propagateBack
    )
    result.weight.copy(elements("weight").asInstanceOf[Tensor[T]])
    result.bias.copy(elements("bias").asInstanceOf[Tensor[T]])
    result
  }

  private def readSequentialModuleWithType[T : ClassTag](
      elements: Table)(implicit ev: TensorNumeric[T]): Sequential[T] = {
    val modules = elements[Table]("modules")
    val result = Sequential[T]()
    for (i <- 1 to modules.length()) {
      result.add(modules(i))
    }
    result
  }
}
