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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.log4j.Logger

trait MiniBatch[T] {
  def toTensorMiniBatch[T](): TensorMiniBatch[T] = {
    this.asInstanceOf[TensorMiniBatch[T]]
  }

  def toArrayTensorMiniBatch[T](): ArrayTensorMiniBatch[T] = {
    this.asInstanceOf[ArrayTensorMiniBatch[T]]
  }

  /**
   * get size of this MiniBatch
   * @return size
   */
  def size(): Int

  /**
   * Check if data's batchSize match labels' batchSize
   */
  def selfCheck(): Unit

  /**
   * Check if batchSize meet the best performance.
   * @param modelNumber number of models
   */
  def performanceCheck(modelNumber: Int): Unit = {
    selfCheck()
    require((size() >= modelNumber) &&
      (size() % modelNumber == 0), "performanceCheck failed: total batch size " +
      s"${size()} should be divided by total core number ${modelNumber}")
    if (size() < modelNumber * 2) {
      MiniBatch.logger.warn("Warning: for better training speed, " +
        "total batch size is recommended to be at least two times of core number" +
        s"${modelNumber}, please tune your batch size accordingly")
    }
  }

  /**
   * Copy this [[MiniBatch]] to an Array of (Activity, Activity)
   * @param dst destination Array
   */
  def copyTo[T](dst: Array[(Activity, Activity)]): Unit = {
    val stackSize = size() / dst.length
    var b = 0
    while (b < dst.length) {
      dst(b) = toActivity(b * stackSize + 1, stackSize)
      b += 1
    }
  }

  /**
   * Slice [[MiniBatch]] to (input, target) with offset and length
   * @param offset offset, counted from 1
   * @param length length
   * @tparam T numeric type
   * @return (input, target)
   */
  def toActivity[T](offset: Int, length: Int): (Activity, Activity)

  /**
   * Transform [[MiniBatch]] to (input, target)
   * @tparam T numeric type
   * @return (input, target)
   */
  def toActivity[T](): (Activity, Activity) = {
    toActivity(1, size())
  }
}

class TensorMiniBatch[T](
      val data: Tensor[T],
      val labels: Tensor[T]) extends MiniBatch[T]{

  override def size(): Int = {
    data.size(1)
  }

  override def selfCheck(): Unit = {
    require(data.size(1) == labels.size(1),
      "MiniBatch selfCheck failed: data and label batch size not match")
  }

  override def toActivity[T](offset: Int, length: Int): (Activity, Activity) = {
    (data.narrow(1, offset, length), labels.narrow(1, offset, length))
  }
}

/**
 * A batch of data feed into the model. The first size is batchsize
 *
 * @param data
 * @param labels
 * @tparam T
 */
class ArrayTensorMiniBatch[T](
      val data: Array[Tensor[T]],
      val labels: Array[Tensor[T]]) extends MiniBatch[T]{

  override def size(): Int = {
    data(0).size(1)
  }

  override def selfCheck(): Unit = {
    require(data(0).size(1) == labels(0).size(1),
      "MiniBatch selfCheck failed: data and label batch size not match")
    var i = 1
    while (i < data.length) {
      require(data(0).size(1) == data(i).size(1),
        "MiniBatch selfCheck failed: batch size of data not match")
      i += 1
    }
    i = 1
    while (i < data.length) {
      require(labels(0).size(1) == labels(i).size(1),
        "MiniBatch selfCheck failed: batch size of label not match")
      i += 1
    }
  }

  override def toActivity[T](offset: Int, length: Int): (Activity, Activity) = {
    if (data.length == 1 && labels.length == 1) {
      (data(0).narrow(1, offset, length),
        labels(0).narrow(1, offset + 1, length))
    } else if (data.length > 1 && labels.length == 1) {
      val newData = T()
      data.foreach(v => newData.insert(v.narrow(1, offset, length)))
      (newData, labels(0).narrow(1, offset, length))
    } else if (data.length == 1 && labels.length > 1) {
      val newLabel = T()
      labels.foreach(v => newLabel.insert(v.narrow(1, offset, length)))
      (data(0).narrow(1, offset, length), newLabel)
    } else {
      val newData = T()
      val newLabel = T()
      data.foreach(v => newData.insert(v.narrow(1, offset, length)))
      labels.foreach(v => newLabel.insert(v.narrow(1, offset, length)))
      (newData, newLabel)
    }
  }
}

object MiniBatch {
  val logger = Logger.getLogger(getClass)

  def apply[T](data: Array[Tensor[T]], labels: Array[Tensor[T]]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](data, labels)
  }

  def apply[T](data: Tensor[T], labels: Tensor[T]): MiniBatch[T] = {
    new TensorMiniBatch[T](data, labels)
  }
}

