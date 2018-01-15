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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Same as ReLU except that the rectifying function f(x) saturates at x = 6
 * ReLU6 is defined as:
 * `f(x) = min(max(0, x), 6)`
 *
 * @param inplace either true = in-place or false = keeping separate state
 */

@SerialVersionUID(8169462538025916360L)
class ReLU6[T: ClassTag](inplace: Boolean = false)
  (implicit ev: TensorNumeric[T])
  extends HardTanh[T](0, 6, inplace) {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    super.updateOutput(input)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    super.updateGradInput(input, gradOutput)
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }
}

object ReLU6 {
  def apply[@specialized(Float, Double) T: ClassTag](
      inplace: Boolean = false)
      (implicit ev: TensorNumeric[T]): ReLU6[T] = {
    new ReLU6[T]()
  }
}
