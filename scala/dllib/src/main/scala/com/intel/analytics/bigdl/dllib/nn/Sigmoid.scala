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

import com.intel.analytics.bigdl.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Applies the Sigmoid function element-wise to the input Tensor,
 * thus outputting a Tensor of the same dimension.
 * Sigmoid is defined as: f(x) = 1 / (1 + exp(-x))
 */
@SerialVersionUID(6855417348268610044L)
class Sigmoid[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).fill(ev.one)
    buffer.resizeAs(input).copy(input).mul(ev.fromType(-1))
    buffer.exp().add(ev.one)
    output.cdiv(buffer)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    updateGradInputInternal(output, gradOutput)
  }

  private[bigdl] def updateGradInputInternal(output: Tensor[T],
                                             gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput).copy(gradOutput)
    buffer.resizeAs(gradOutput)
    buffer.fill(ev.one).sub(output)
    gradInput.cmul(output).cmul(buffer)
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    buffer.set()
    this
  }
}

object Sigmoid {
  def apply[T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Sigmoid[T] = {
    new Sigmoid[T]()
  }
}
