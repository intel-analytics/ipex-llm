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
import com.intel.analytics.bigdl.utils.{T, Table}


/**
 * A interface for MiniBatch.
 * A MiniBatch contains a few samples.
 *
 * @tparam T Numeric type
 */
trait MiniBatch[T] extends Serializable{
  /**
   * Get the number of samples in this MiniBatch
   * @return size How many samples in this MiniBatch
   */
  def size(): Int

  /**
   * Slice this MiniBatch to a smaller MiniBatch with offset and length
   * @param offset offset, counted from 1
   * @param length length
   * @return A smaller MiniBatch
   */
  def slice(offset: Int, length: Int): MiniBatch[T]

  /**
   * Get input in this MiniBatch.
   * @return input Activity
   */
  def getInput(): Activity

  /**
   * Get target in this MiniBatch
   * @return target Activity
   */
  def getTarget(): Activity

  @deprecated("Old interface", "0.2.0")
  def data(): Tensor[T] = {
    require(this.isInstanceOf[TensorMiniBatch[T]], "Deprecated method," +
      " Only support TensorMiniBatch.")
    this.asInstanceOf[TensorMiniBatch[T]].input
  }

  @deprecated("Old interface", "0.2.0")
  def labels(): Tensor[T] = {
    require(this.isInstanceOf[TensorMiniBatch[T]], "Deprecated method," +
      " Only support TensorMiniBatch.")
    this.asInstanceOf[TensorMiniBatch[T]].input
  }
}

/**
 * A MiniBatch with [[Tensor]] input and [[Tensor]] target.
 * The size of first dimension in input and target should be the mini-batch size.
 *
 * @param input input Tensor
 * @param target target Tensor
 * @tparam T Numeric type
 */
class TensorMiniBatch[T](
      val input: Tensor[T],
      val target: Tensor[T]) extends MiniBatch[T]{
  require(input.size(1) == target.size(1))

  override def size(): Int = {
    input.size(1)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    MiniBatch(input.narrow(1, offset, length), target.narrow(1, offset, length))
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    target
  }
}

/**
 * A MiniBatch with [[com.intel.analytics.bigdl.utils.Table]] input and [[Tensor]] target.
 * The size of first dimension in input's first tensor and target is the mini-batch size.
 *
 * @param input input Table
 * @param target target Tensor
 * @tparam T Numeric type
 */
class ArrayTensorMiniBatch[T](
      val input: Table,
      val target: Tensor[T]) extends MiniBatch[T]{

  override def size(): Int = {
    input[Tensor[T]](1).size(1)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    val t = T()
    var b = 1
    while(b <= input.length()) {
      t(b) = input[Tensor[T]](b).narrow(1, offset, length)
      b += 1
    }
    MiniBatch(t, target.narrow(1, offset, length))
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    target
  }
}

/**
 * TensorMiniBatch without target.
 * @param input input tensor of this MiniBatch
 * @tparam T Numeric type
 */
class UnlabeledTensorMiniBatch[T](
      val input: Tensor[T]) extends MiniBatch[T]{

  override def size(): Int = {
    input.size(1)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    MiniBatch(input.narrow(1, offset, length))
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    throw new UnsupportedOperationException("UnlabeledTensorMiniBatch: no target in this MiniBatch")
  }
}

/**
 * ArrayTensorMiniBatch without target.
 * @param input input table of this MiniBatch
 * @tparam T Numeric type
 */
class UnlabeledArrayTensorMiniBatch[T](
      val input: Table) extends MiniBatch[T]{

  override def size(): Int = {
    input[Tensor[T]](1).size(1)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    val t = T()
    var b = 1
    while(b <= input.length()) {
      t(b) = input[Tensor[T]](b).narrow(1, offset, length)
      b += 1
    }
    MiniBatch(t)
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    throw new UnsupportedOperationException("No target in this UnlabeledArrayTensorMiniBatch")
  }
}

object MiniBatch {
  def apply[T](input: Tensor[T], target: Tensor[T]): MiniBatch[T] = {
    new TensorMiniBatch[T](input, target)
  }

  def apply[T](input: Table, target: Tensor[T]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](input, target)
  }

  def apply[T](input: Tensor[T]): MiniBatch[T] = {
    new UnlabeledTensorMiniBatch[T](input)
  }

  def apply[T](input: Table): MiniBatch[T] = {
    new UnlabeledArrayTensorMiniBatch[T](input)
  }
}

