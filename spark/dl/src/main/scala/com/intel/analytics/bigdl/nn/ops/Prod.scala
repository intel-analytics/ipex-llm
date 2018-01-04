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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}

import scala.reflect.ClassTag

class Prod[T: ClassTag](
  axis: Int = 1,
  keepDim: Boolean = false)
(implicit ev: TensorNumeric[T]) extends Operation[Tensor[_], Tensor[_], T] {
  private def getPositiveDimension(input: Tensor[_]): Int = {
    var dimension = this.axis
    if (dimension < 0) {
      dimension = input.dim() + dimension + 1
    }

    require(input.dim() >= dimension, "dimension exceeds input dimensions")
    dimension
  }

  def updateOutput(input: Tensor[_]): Tensor[_] = {
    val dimension = getPositiveDimension(input)
    if (output.getType() != input.getType()) {
      output = input.emptyInstance()
    }
    output.asInstanceOf[Tensor[NumericWildCard]]
      .prod(input.asInstanceOf[Tensor[NumericWildCard]], dimension)

    if (output.nDimension() > 1 && !keepDim) {
      output.squeeze(dimension)
    }

    output
  }
}

object Prod {
  def apply[T: ClassTag](axis: Int, keepDim: Boolean = false)
    (implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](
    new Prod(axis = axis, keepDim = keepDim))
}
