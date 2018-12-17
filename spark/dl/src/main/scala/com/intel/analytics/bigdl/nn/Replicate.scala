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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Replicate repeats input `nFeatures` times along its `dim` dimension
 *
 * Notice: No memory copy, it set the stride along the `dim`-th dimension to zero.
 *
 * @param nFeatures replicate times.
 * @param dim dimension to be replicated.
 * @param nDim specify the number of non-batch dimensions.
 */

@SerialVersionUID( - 7255265230723863741L)
class Replicate[T: ClassTag](
  val nFeatures : Int,
  val dim : Int = 1,
  val nDim : Int = Int.MaxValue)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(dim > 0, "Can only replicate across positive integer dimensions. " +
    s"The number of dimensions is $dim.")

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(dim <= input.dim() + 1,
      s"Not enough input dimensions to replicate along dimension $dim.")

    val batchOffset = if (input.dim() > nDim) 1 else 0
    val rDim = dim + batchOffset
    val size = new Array[Int](input.dim() + 1)
    size(rDim - 1) = nFeatures
    val stride = new Array[Int](input.dim() + 1)
    stride(rDim - 1) = 0
    var i = 1
    while (i <= input.dim()) {
      val offset = if (i >= rDim) 1 else 0
      size(i + offset - 1) = input.size(i)
      stride(i + offset - 1) = input.stride(i)
      i += 1
    }
    output.set(input.storage(), input.storageOffset(), size, stride)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val batchOffset = if (input.dim() > nDim) 1 else 0
    val rDim = dim + batchOffset
    val size = new Array[Int](input.dim() + 1)
    size(rDim - 1) = 1
    var i = 1
    while (i <= input.dim()) {
      val offset = if (i >= rDim) 1 else 0
      size(i + offset - 1) = input.size(i)
      i += 1
    }
    gradInput.view(size).sum(gradOutput, rDim)

    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($nFeatures, $dim${if (nDim != Int.MaxValue) ", " + nDim else ""})"
  }
}

object Replicate {
  def apply[@specialized(Float, Double) T: ClassTag](
      nFeatures : Int,
      dim : Int = 1,
      nDim : Int = Int.MaxValue)(implicit ev: TensorNumeric[T]) : Replicate[T] = {
    new Replicate[T](nFeatures, dim, nDim)
  }
}
