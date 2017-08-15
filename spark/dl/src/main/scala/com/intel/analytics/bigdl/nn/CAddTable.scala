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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect._

/**
 * Merge the input tensors in the input table by element wise adding them together. The input table
 * is actually an array of tensor with same size.
 * @param inplace reuse the input memory
 * @param ev numeric operator
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(7959261460060075605L)
class CAddTable[T: ClassTag](val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    if (inplace) {
      output.set(input[Tensor[T]](1))
    } else {
      output.resizeAs(input[Tensor[T]](1)).copy(input[Tensor[T]](1))
    }
    var i = 2
    while (i <= input.length()) {
      output.add(input[Tensor[T]](i))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    var i = 1
    while (i <= input.length()) {
      if (i > gradInput.length) gradInput.insert(i, Tensor[T]().resizeAs(input(1)))
      if (inplace) {
        gradInput[Tensor[T]](i).set(gradOutput)
      } else {
        gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
      }
      i += 1
    }
    i = input.length + 1
    while (i <= gradInput.length) {
      gradInput.remove(i)
    }
    gradInput
  }
}


object CAddTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : CAddTable[T] = {
    new CAddTable[T](inplace)
  }
}
