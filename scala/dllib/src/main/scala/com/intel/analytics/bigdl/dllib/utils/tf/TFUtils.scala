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
package com.intel.analytics.bigdl.utils.tf

import java.nio.{ByteBuffer, ByteOrder}

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.SparkContext
import org.tensorflow.framework.{DataType, TensorProto}

import scala.collection.JavaConverters._


object TFUtils {
  import TFTensorNumeric.NumericByteString

  /**
   * Read the local files into RDD and write to specified path in HDFS
   */
  def saveToHDFS(localFiles: Seq[String], outputPath: String,
                 numPartitions: Int, sc: SparkContext): Unit = {
    val result = localFiles.flatMap { file =>
      val iter = TFRecordIterator(new java.io.File(file))
      iter
    }
    sc.parallelize(result, numPartitions)
      .map(bytes => (new BytesWritable(bytes), NullWritable.get()))
      .saveAsNewAPIHadoopFile[TFRecordOutputFormat](outputPath)
  }

  private def parseTensorFromContent(
    dataType: DataType, content: Array[Byte], shape: Array[Int], endian: ByteOrder) = {
    dataType match {
      case DataType.DT_FLOAT =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer.asFloatBuffer()
        val tmp = new Array[Float](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor[Float](tmp, shape)
      case DataType.DT_DOUBLE =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer.asDoubleBuffer()
        val tmp = new Array[Double](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor[Double](tmp, shape)
      case DataType.DT_INT32 =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer.asIntBuffer()
        val tmp = new Array[Int](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor[Int](tmp, shape)
      case DataType.DT_INT64 =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer.asLongBuffer()
        val tmp = new Array[Long](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor[Long](tmp, shape)
      case DataType.DT_INT8 =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer
        val tmp = new Array[Int](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(tmp, shape)
      case DataType.DT_UINT8 =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer
        val tmp = new Array[Int](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j) & 0xff
          j += 1
        }
        Tensor(tmp, shape)
      case DataType.DT_INT16 =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer.asShortBuffer()
        val tmp = new Array[Int](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(tmp, shape)
      case DataType.DT_UINT16 =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer.asShortBuffer()
        val tmp = new Array[Int](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j) & 0xffff
          j += 1
        }
        Tensor(tmp, shape)
      case DataType.DT_BOOL =>
        val buffer = ByteBuffer.wrap(content)
        buffer.order(endian)
        val params = buffer
        val tmp = new Array[Boolean](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = if (params.get(j) == 0) false else true
          j += 1
        }
        Tensor(tmp, shape)
      case t => throw new IllegalArgumentException(s"DataType: $t not supported yet")
    }
  }

  private def parseTensorFromField(
    tfTensor: TensorProto, shape: Array[Int], endian: ByteOrder) = {
    tfTensor.getDtype match {
      case DataType.DT_FLOAT =>
        val tmp = tfTensor.getFloatValList.asScala.map(_.toFloat).toArray
        Tensor[Float](tmp, shape)
      case DataType.DT_DOUBLE =>
        val tmp = tfTensor.getDoubleValList.asScala.map(_.toDouble).toArray
        Tensor[Double](tmp, shape)
      case DataType.DT_INT32 =>
        val tmp = tfTensor.getIntValList.asScala.map(_.toInt).toArray
        Tensor[Int](tmp, shape)
      case DataType.DT_INT64 =>
        val tmp = tfTensor.getInt64ValList.asScala.map(_.toLong).toArray
        Tensor[Long](tmp, shape)
      case DataType.DT_BOOL =>
        val tmp = tfTensor.getBoolValList.asScala.map(_.booleanValue()).toArray
        Tensor[Boolean](tmp, shape)
      case DataType.DT_STRING =>
        val tmp = tfTensor.getStringValList.asScala.toArray
        Tensor[ByteString](tmp, shape)
      case DataType.DT_INT8 =>
        val tmp = tfTensor.getIntValList.asScala.map(_.toInt).toArray
        Tensor(tmp, shape)
      case DataType.DT_UINT8 =>
        val tmp = tfTensor.getIntValList.asScala.map(_.toInt).toArray
        Tensor(tmp, shape)
      case DataType.DT_INT16 =>
        val tmp = tfTensor.getIntValList.asScala.map(_.toInt).toArray
        Tensor(tmp, shape)
      case DataType.DT_UINT16 =>
        val tmp = tfTensor.getIntValList.asScala.map(_.toInt).toArray
        Tensor(tmp, shape)
      case t => throw new IllegalArgumentException(s"DataType: $t not supported yet")
    }
  }

  /**
   * convert tensorflow tensorProto to BigDL Tensor
   */
  def parseTensor(tfTensor: TensorProto, endian: ByteOrder): Tensor[_] = {
    val shape = tfTensor.getTensorShape.getDimList.asScala.map(_.getSize.toInt).toArray
    if (tfTensor.getTensorContent.isEmpty) {
      parseTensorFromField(tfTensor, shape, endian)
    } else {
      parseTensorFromContent(tfTensor.getDtype,
        tfTensor.getTensorContent.toByteArray, shape, endian)
    }
  }
}
