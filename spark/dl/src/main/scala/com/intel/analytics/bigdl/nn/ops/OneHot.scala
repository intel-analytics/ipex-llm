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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * OneHot operation returns a one-hot tensor
 *
 * The input contains 4 elements which are `indices`, `depth`, `onValue` and `offValue`*[]:

 * The locations represented by indices in `indices` take value `onValue`,
 * while all other locations take value `offValue`.
 *
 * `onValue` and `offValue` must have matching data types.
 * If dtype is also provided, they must be the same data type as specified by `D`.
 *
 * If on_value is not provided, it will default to the value 1 with type dtype
 *
 * If off_value is not provided, it will default to the value 0 with type dtype
 *
 * If the input indices is rank N, the output will have rank N+1.
 * The new axis is created at dimension axis (default: the new axis is appended at the end).
 *
 * If indices is a scalar the output shape will be a vector of length depth
 *
 * If indices is a vector of length features, the output shape will be:
 * features x depth if axis == -1
 * depth x features if axis == 0
 *
 * If indices is a matrix (batch) with shape [batch, features], the output shape will be:
 *
 * batch x features x depth if axis == -1
 * batch x depth x features if axis == 1
 * depth x batch x features if axis == 0
 *
 * @param axis The new axis is created at dimension axis
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 * @tparam D Numeric type. Output tensor numeric type. Only support float/double now
 */
class OneHot[T: ClassTag, D: ClassTag](
  axis: Int
)(implicit ev: TensorNumeric[T], ev1: TensorNumeric[D]) extends Operation[Table, Tensor[D], T] {
  output = Activity.allocate[Tensor[D], D]()
  def updateOutput(input: Table): Tensor[D] = {
    val indices = input[Tensor[Long]](1)
    val depth = input[Tensor[Int]](2).value()
    val onValue = if (!input.contains(3)) ev1.one else input[Tensor[D]](3).value()
    val offValue = if (!input.contains(4)) ev1.zero else input[Tensor[D]](4).value()

    require(input[Tensor[_]](3).getType() == input[Tensor[_]](4).getType(),
    "onValue must have the same type as offValue")

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
    output.apply1(x => offValue)

    if (size.length == 2) {
      i = 1
      while (i <= size(0)) {
        j = 1
        while (j <= size(1)) {
          val index = (indices(Array(i, j)) + 1).toInt
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
        val index = (indices(Array(i)) + 1).toInt
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

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev1))
  }
}

object OneHot {
  def apply[T: ClassTag, D: ClassTag](
    axis: Int
  )
    (implicit ev: TensorNumeric[T], ev1: TensorNumeric[D]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](
    new OneHot[T, D](
      axis = axis
    ))
}
