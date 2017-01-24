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
 * Represent a MiniBatch
 */
abstract class MiniBatch[T: ClassTag](val data: Activity, val labels: Activity)
  (implicit ev: TensorNumeric[T]) extends Serializable {

  def size(): Int

  def get: (Activity, Activity)

  /**
   * To select a sub minibatch starting from index
   * The TensorMinibatch calls the Tensor.narrow method
   * The TableMinibatch calls the NarrowTable layer
   * Will return a subTensor or a subTable of Minibatch
   *
   * @param dim the direction to select the sub minibatch
   *            for Table data, the dim is set to be 1
   * @param index the start point of the minibatch
   * @param size the length of the sub minibatch
   */
  def narrow(dim: Int, index: Int, size: Int): MiniBatch[T]
}

/**
 * A TensorBatch of data feed into the model. The first dimension is batchsize
 *
 * @param data
 * @param labels
 * @tparam T
 */
class TensorMiniBatch[T: ClassTag](data: Tensor[T], labels: Tensor[T])
  (implicit ev: TensorNumeric[T]) extends MiniBatch[T](data, labels) {

  require(data.size(1) == labels.size(1),
    "data and label batch size do not match")

  def this (pair: (Tensor[T], Tensor[T]))
  (implicit ev: TensorNumeric[T]) {
    this(pair._1, pair._2)
  }

  override def size(): Int = data.size(1)

  override def get: (Tensor[T], Tensor[T]) = (data, labels)

  override def narrow(dim: Int, index: Int, size: Int): TensorMiniBatch[T] = {
    new TensorMiniBatch[T](data.narrow(1, index, size), labels.narrow(dim, index, size))
  }
}

object TensorMiniBatch {
  def apply[T: ClassTag](
    data: Tensor[T],
    labels: Tensor[T])
  (implicit ev: TensorNumeric[T]): TensorMiniBatch[T]
  = new TensorMiniBatch[T](data, labels)

  def apply[T: ClassTag](pair: (Tensor[T], Tensor[T]))
  (implicit ev: TensorNumeric[T]): TensorMiniBatch[T]
  = new TensorMiniBatch[T](pair)
}

/**
 * A TableBatch of data feed into the model. The first dimension is batchsize
 *
 * @param data
 * @param labels
 * @tparam T
 */
class TableMiniBatch[T: ClassTag](data: Table, labels: Table, length: Int)
  (implicit ev: TensorNumeric[T]) extends MiniBatch[T](data, labels) {

  require(data.length >= length && labels.length >= length,
    "data and label batch size do not match")

  def this (triple: (Table, Table, Int))
           (implicit ev: TensorNumeric[T]) {
    this(triple._1, triple._2, triple._3)
  }

  override def size(): Int = length

  override def get: (Table, Table) = {
    (NarrowTable[T](1, length).updateOutput(data),
      NarrowTable[T](1, length).updateOutput(labels))
  }

  override def narrow(dim: Int, index: Int, size: Int): TableMiniBatch[T] = {
    TableMiniBatch[T](NarrowTable[T](index, size).updateOutput(data),
          NarrowTable[T](index, size).updateOutput(labels), size)
  }
}

object TableMiniBatch {
  def apply[T: ClassTag](
    data: Table,
    labels: Table,
    length: Int)
  (implicit ev: TensorNumeric[T]): TableMiniBatch[T]
  = new TableMiniBatch[T](data, labels, length)

  def apply[T: ClassTag](pair: (Table, Table, Int))
  (implicit ev: TensorNumeric[T]): TableMiniBatch[T]
  = new TableMiniBatch[T](pair)
}
