/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Applies the SoftMax function to an n-dimensional input Tensor, rescaling them so that the
 * elements of the n-dimensional output Tensor lie in the range (0, 1) and sum to 1.
 * Softmax is defined as: f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
 * where shift = max_i(x_i).
 * Currently only support apply softmax normalization to the last dim.
 */
private[zoo] class InternalSoftMax[T: ClassTag]()
   (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val sizes = input.size()
    val shift = input.max(dim)._1

    val shiftedInput = input.clone().sub(shift.expand(sizes).contiguous())
    val exp = shiftedInput.exp()

    val sum = exp.sum(dim)
    output = exp.div(sum.expand(sizes).contiguous())

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val sum = (output.clone().cmul(gradOutput)).sum(dim)
    gradInput = output.clone().cmul(gradOutput - sum.expand(input.size()))
    gradInput
  }
}

private[zoo] object InternalSoftMax{
  def apply[T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : InternalSoftMax[T] = {
    new InternalSoftMax[T]()
  }
}
