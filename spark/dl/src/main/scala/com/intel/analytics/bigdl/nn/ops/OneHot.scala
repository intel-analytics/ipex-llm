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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class OneHot[T: ClassTag](
  axis: Int
)(implicit ev: TensorNumeric[T]) extends Operation[Table, T] {
  def updateOutput(input: Table): Tensor[T] = {
    val indices = input[Tensor[Int]](1)
    val depth = input[Int](2)
    val onValue = input[T](3)
    val offValue = input[T](4)

    val size: Array[Int] = indices.size()
    require(indices.dim() <= 2 && indices.dim() > 0,
      "the dimension of input must be less than or equal to 2")
    val newSize: Array[Int] = new Array(size.length + 1)

    val realAxis = if (axis == -1) newSize.length - 1

    var i = 0
    var j = 0
    while (i < newSize.length) {
      if (realAxis == i) {
        newSize(i) = depth
      } else {
        newSize(i) = size(j)
        j += 1
      }

      i += 1
    }

    output.resize(newSize)
    output.copy(output.apply1(x => offValue))

    if (size.length == 2) {
      i = 1
      while (i <= size(0)) {
        j = 1
        while (j <= size(1)) {
          val index = indices(Array(i, j)) + 1
          if (index > 0) {
            if (realAxis == 0) {
              output.setValue(index, i, j, onValue)
            } else if (realAxis == 1) {
              output.setValue(i, index, j, onValue)
            } else if (realAxis == 2) {
              output.setValue(i, j, index, onValue)
            }
          }
          j += 1
        }
        i += 1
      }
    } else {
      i = 1
      while (i <= size(0)) {
        val index = indices(Array(i)) + 1
        if (index > 0) {
          if (realAxis == 0) {
            output.setValue(index, i, onValue)
          } else if (realAxis == 1) {
            output.setValue(i, index, onValue)
          }
        }
        i += 1
      }
    }
    output
  }
}

object OneHot {
  def apply[T: ClassTag](
    axis: Int
  )
    (implicit ev: TensorNumeric[T]): Operation[Table, T]
  = ModuleToOperation[Table, T](
    new OneHot(
      axis = axis
    ))
}
