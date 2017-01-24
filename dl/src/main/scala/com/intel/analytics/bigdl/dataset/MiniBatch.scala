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

import com.intel.analytics.bigdl.nn.NarrowTable
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * A batch of data feed into the model. The first size is batchsize
 *
 * @param data
 * @param labels
 * @tparam T
 */
case class MiniBatch[T: ClassTag](data: Activity, labels: Activity)
  (implicit ev: TensorNumeric[T]) extends Serializable {

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

  def get: (Activity, Activity) = (data, labels)

  /**
   * To select a sub minibatch starting from index
   * The Tensor data calls the Tensor.narrow method
   * The Table data calls the NarrowTable layer
   * Will return a subTensor or a subTable
   *
   * @param dim the direction to select the sub minibatch
   *            for Table data, the dim is set to be 1
   * @param index the start point of the minibatch
   * @param size the length of the sub minibatch
   */
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
