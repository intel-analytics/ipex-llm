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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.math.tanh
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

/**
 * Applies the Tanh function element-wise to the input Tensor,
 * thus outputting a Tensor of the same dimension.
 * Tanh is defined as f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)).
 */
@SerialVersionUID(9062199894710333035L)
class Tanh[@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.map(input, (_, inputVal) => ev.fromType[Double](tanh(ev.toType[Double](inputVal))))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
    gradInput.copy(gradOutput)
    gradInput.map(output, (gradValue, outputValue) => ev.times(
      gradValue, ev.minus(ev.fromType[Int](1), ev.times(outputValue, outputValue))))
    gradInput
  }
}


object Tanh {
  def apply[T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Tanh[T] = {
    new Tanh[T]()
  }
}

