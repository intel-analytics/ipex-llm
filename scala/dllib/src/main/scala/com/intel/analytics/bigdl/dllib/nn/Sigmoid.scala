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
 * Applies the Sigmoid function element-wise to the input Tensor,
 * thus outputting a Tensor of the same dimension.
 */

@SerialVersionUID(6855417348268610044L)
class Sigmoid[@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T]  {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.map(input, (_, i) => ev.divide(ev.fromType[Int](1), ev.plus(ev.fromType[Int](1),
      ev.exp(ev.negative(i)))))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    gradInput.copy(gradOutput)
    gradInput.map(output, (g, z) => ev.times(ev.times(g, ev.minus(ev.fromType[Int](1), z)), z))
    gradInput
  }

  override def toString(): String = {
    s"nn.Sigmoid"
  }
}

object Sigmoid {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Sigmoid[T] = {
    new Sigmoid[T]()
  }
}
