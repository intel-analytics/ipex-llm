/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.dataset

import java.nio.ByteBuffer
import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.image.LabeledBGRImage
import com.intel.analytics.bigdl.dataset.text.LabeledSentence
import com.intel.analytics.bigdl.nn.NarrowTable
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table

import scala.collection.Iterator
import scala.reflect.ClassTag

/**
 * Represent an image
 */
abstract class Image extends Serializable {
  def width(): Int

  def height(): Int

  def content: Array[Float]
}

 /**
  * Represent a sentence
  */
abstract class Sentence[T] extends Serializable {
  def dataLength(): Int

  def data(): Array[T]
}

/**
 * Represent a local file path of an image file
 *
 * @param path
 */
class LocalImagePath(val path : Path)

/**
 * Represent a local file path of a hadoop sequence file
 *
 * @param path
 */
case class LocalSeqFilePath(val path: Path)


 /**
  * Sample, bundling input and target
  *
  * @param featureTensor
  * @param labelTensor
  * @tparam T
  */

class Sample[T: ClassTag] (
    protected var featureTensor: Tensor[T],
    protected var labelTensor: Tensor[T])
    (implicit ev: TensorNumeric[T]) extends Serializable {

  def this()(implicit ev: TensorNumeric[T]) = this(Tensor[T](), Tensor[T]())

  def copy(featureData: Array[T],
           labelData: Array[T],
           featureSize: Array[Int],
           labelSize: Array[Int]): Sample[T] = {
    featureTensor.set(Storage[T](featureData), 1, featureSize)
    labelTensor.set(Storage[T](labelData), 1, labelSize)
    this
  }

  def extractFeature(storage: Array[T], offset: Int, length: Int): Unit = {
    require(offset + length <= storage.length, "index out of boundary")
    require(length <= featureTensor.storage().array().length, "length too long for feature")
    ev.getType() match {
      case DoubleType => Array.copy(featureTensor.storage().array
          .asInstanceOf[Array[Double]],
        featureTensor.storageOffset() - 1, storage
          .asInstanceOf[Array[Double]], offset, length)
      case FloatType => System.arraycopy(featureTensor.storage().array
        .asInstanceOf[Array[Float]],
        featureTensor.storageOffset() - 1, storage
        .asInstanceOf[Array[Float]], offset, length)
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  def extractLabel(storage: Array[T], offset: Int, length: Int): Unit = {
    require(offset + length <= storage.length, "index out of boundary")
    require(length <= labelTensor.storage().array().length, "length too long for label")
    ev.getType() match {
      case DoubleType => Array.copy(labelTensor.storage().array
        .asInstanceOf[Array[Double]],
        labelTensor.storageOffset() - 1, storage
        .asInstanceOf[Array[Double]], offset, length)
      case FloatType => Array.copy(labelTensor.storage().array
        .asInstanceOf[Array[Float]],
        labelTensor.storageOffset() - 1, storage
        .asInstanceOf[Array[Float]], offset, length)
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
  }

  def copy(other: Sample[T]): Sample[T] = {
    this.featureTensor.resizeAs(other.featureTensor).copy(other.featureTensor)
    this.labelTensor.resizeAs(other.labelTensor).copy(other.labelTensor)
    this
  }

  def feature(): Tensor[T] = featureTensor

  def label(): Tensor[T] = labelTensor

  override def clone(): Sample[T] = {
    new Sample[T]().copy(this)
  }
}

object Sample {
  def apply[@specialized(Float, Double) T: ClassTag]
  (featureTensor: Tensor[T], labelTensor: Tensor[T])
  (implicit ev: TensorNumeric[T]) : Sample[T] = {
    new Sample[T](featureTensor, labelTensor)
  }
  def apply[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]) : Sample[T] = {
    new Sample[T]()
  }
}


/**
 * Represent a label
 *
 * @tparam T
 */
trait Label[T] {
  def setLabel(label: T): this.type
  def label(): T
}

/**
 * A batch of data feed into the model. The first size is batchsize
 * @param data
 * @param labels
 * @tparam T
 */
case class MiniBatch[T: ClassTag](data: Activity, labels: Activity)
  (implicit ev: TensorNumeric[T]) {

  val _dataLength = dataLength()
  val _labelLength = labelLength()

  require(_dataLength == _labelLength,
    "data and label batch size not match")

  def this (pair: (Activity, Activity))
  (implicit ev: TensorNumeric[T]) {
    this(pair._1, pair._2)
  }

  def unapply[@specialized(Float, Double) T: ClassTag]
  (batch: MiniBatch[T])
  (implicit ev: TensorNumeric[T]): Option[(Activity, Activity)] =
    Some((batch.data, batch.labels))

  private def dataLength(): Int = {
    data match {
      case tensor: Tensor[T] => tensor.size(1)
      case table: Table => table.length
      case _ => throw new IllegalArgumentException("MiniBatch only supports Table or Tensor")
    }
  }

  private def labelLength(): Int = {
    labels match {
      case tensor: Tensor[T] => tensor.size(1)
      case table: Table => table.length
      case _ => throw new IllegalArgumentException("MiniBatch only supports Table or Tensor")
    }
  }

  def size(): Int = _dataLength

  def narrow(dim: Int, index: Int, size: Int): MiniBatch[T] = {
    data match {
      case tensor: Tensor[T] =>
        MiniBatch[T](tensor.narrow(1, index, size), labels.toTensor[T].narrow(dim, index, size))
      case table: Table =>
        MiniBatch[T](NarrowTable[T](index, size).updateOutput(table),
          NarrowTable[T](index, size).updateOutput(labels.toTable))
      case _ => throw new IllegalArgumentException("MiniBatch only supports Table or Tensor")
    }
  }
}

/**
 * A byte array and a label. It can contain anything.
 * @param data
 * @param label
 */
case class ByteRecord(data: Array[Byte], label: Float)
