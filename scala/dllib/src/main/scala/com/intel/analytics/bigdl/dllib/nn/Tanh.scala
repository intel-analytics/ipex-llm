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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.math.tanh
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Applies the Tanh function element-wise to the input Tensor,
 * thus outputting a Tensor of the same dimension.
 * Tanh is defined as f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)).
 */
@SerialVersionUID(9062199894710333035L)
class Tanh[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.tanh(input)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    updateGradInputInternal(output, gradOutput)
  }

  private[bigdl] def updateGradInputInternal(output: Tensor[T],
                                            gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
    buffer.resizeAs(output)
    buffer.pow(output, ev.fromType(2)).cmul(gradOutput)
    gradInput.sub(gradOutput, buffer)
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    buffer.set()
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}


object Tanh {
  def apply[T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Tanh[T] = {
    new Tanh[T]()
  }
}

