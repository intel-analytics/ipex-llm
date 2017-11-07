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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}

import scala.reflect.ClassTag

/**
 * This operation extracts a slice of size size from a tensor
 * input starting at the location specified by begin.
 * The slice size is represented as a tensor shape, where size[i] is
 * the number of elements of the 'i'th dimension of input that you want to slice
 * The starting location (begin) for the slice is represented as an offset in each
 * dimension of input.
 * In other words, begin[i] is the offset into the 'i'th dimension of input that you
 * want to slice from.
 *
 * @param begin zero-based
 * @param size one-based
 * @tparam T Numeric type. Only support float/double now
 */
class Slice[T: ClassTag](
  begin: Array[Int],
  size: Array[Int])
  (implicit ev: TensorNumeric[T]) extends Operation[Tensor[_], Tensor[_], T] {

  def updateOutput(input: Tensor[_]): Tensor[_] = {
    require(begin.length == size.length && begin.length == input.dim(),
      "the length of `begin`, `size` and the dimension of input should be the same")

    var outputNarrow = input
    var i = 0
    while (i < begin.length) {
      val realSize = if (size(i) == -1) input.size(i + 1) - begin(i) else size(i)
      outputNarrow = outputNarrow.narrow(i + 1, begin(i) + 1, realSize)
      i += 1
    }
    if (output.getType() != input.getType()) {
      output = input.emptyInstance()
    }
    output.resizeAs(outputNarrow)
    output.asInstanceOf[Tensor[NumericWildcard]]
      .copy(outputNarrow.asInstanceOf[Tensor[NumericWildcard]])

    output
  }
}

object Slice {
  def apply[T: ClassTag](
    begin: Array[Int],
    size: Array[Int])
    (implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](
    new Slice(begin = begin, size = size))
}
