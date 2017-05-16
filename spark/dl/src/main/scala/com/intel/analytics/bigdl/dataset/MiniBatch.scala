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

/**
 * A interface for MiniBatch.
 * A MiniBatch contains a few samples.
 *
 * @tparam T Numeric type
 */
trait MiniBatch[T] {
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

object MiniBatch {
  def apply[T](input: Tensor[T], target: Tensor[T]): MiniBatch[T] = {
    new TensorMiniBatch[T](input, target)
  }
}

