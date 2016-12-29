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
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

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
abstract class Sentence extends Serializable {
  def length(): Int

  def content: Array[Float]
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
  */

class Sample[T](
    protected var featureTensor: Tensor[T],
    protected var labelTensor: Tensor[T]) {

  def this() = this(Tensor[T](), Tensor[T]())

  def copy(featureData: Array[T],
           labelData: Array[T],
           featureSize: Array[Int],
           labelSize: Array[Int]): Sample[T] = {
    featureTensor.set(Storage[T](featureData), 1, featureSize)
    labelTensor.set(Storage[T](labelData), 1, labelSize)
    this
  }

  def copyToFeature(storage: Array[T], offset: Int, length: Int): Unit = {
    require(offset + length <= storage.length, "index out of boundary")
    var i = 0
    while (i < length) {
      storage(offset + i) = featureTensor.storage()(i)
      i += 1
    }
  }

  def copyToLabel(storage: Array[T], offset: Int, length: Int): Unit = {
    require(offset + length <= storage.length, "index out of boundary")
    var i = 0
    while (i < length) {
      storage(offset + i) = labelTensor.storage()(i)
      i += 1
    }
  }

  def copy(other: Sample[T]): Sample[T] = {
    featureTensor.copy(other.featureTensor)
    labelTensor.copy(other.labelTensor)
    this
  }

  def getFeature(): Tensor[T] = featureTensor

  def getLabel(): Tensor[T] = labelTensor

  override def clone(): Sample[T] = {
    new Sample().copy(this)
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

case class MiniBatch[T](data: Tensor[T], labels: Tensor[T])

case class ByteRecord(data: Array[Byte], label: Float)
