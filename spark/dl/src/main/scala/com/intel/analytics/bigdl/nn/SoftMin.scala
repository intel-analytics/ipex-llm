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

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Applies the SoftMin function to an n-dimensional input Tensor, rescaling them so that the
 * elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1.
 * Softmin is defined as: f_i(x) = exp(-x_i - shift) / sum_j exp(-x_j - shift)
 * where shift = max_i(-x_i).
 */
@SerialVersionUID(- 8738615460960887232L)
class SoftMin[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient
  private var results: Array[Future[Unit]] = null
  @transient
  private var minInput : Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val (nFrame, stride) = if (input.nDimension() == 1) {
      (1, 1)
    } else if (input.nDimension() == 2) {
      (input.size(1), 1)
    } else if (input.nDimension() == 3) {
      (1, input.size(2) * input.size(3))
    } else {
      (input.size(1), input.size(3) * input.size(4))
    }
    if (results == null || results.length != nFrame * stride) {
      results = new Array[Future[Unit]](nFrame * stride)
    }
    output.resizeAs(input)
    if (null == minInput) {
      minInput = input.clone().mul(ev.fromType[Int](-1))
    } else {
      minInput.resizeAs(input).copy(input).mul(ev.fromType[Int](-1))
    }
    SoftMax.updateOutput[T](minInput, output, results)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output)
    SoftMax.updateGradInput[T](minInput, gradOutput, gradInput, output, results)
    gradInput.mul(ev.fromType[Int](-1))
    gradInput
  }
}

object SoftMin {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : SoftMin[T] = {
    new SoftMin[T]()
  }
}


