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

class ActivityRegularization[T: ClassTag](val l1: Double, val l2: Double)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  var loss: T = ev.fromType(0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    loss = ev.plus(ev.times(input.norm(1), ev.fromType(l1)), // l1
      ev.times(ev.pow(input.norm(2), ev.fromType(2)), ev.fromType(l2))) // l2

    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
      .copy(input).sign().mul(ev.fromType(l1)) // l1
      .add(input.mul(ev.fromType(2 * l2))) // l2
      .add(gradOutput) // add all the gradients of branches

    gradInput
  }
}

object ActivityRegularization {
  def apply[T: ClassTag](l1: Double, l2: Double)(
    implicit ev: TensorNumeric[T]): ActivityRegularization[T] = {
    new ActivityRegularization[T](l1, l2)
  }
}
