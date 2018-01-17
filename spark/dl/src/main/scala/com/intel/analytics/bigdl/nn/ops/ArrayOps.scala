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
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * This operation computes the inverse of an index permutation. It takes a 1-D integer tensor x,
 * which represents the indices of a zero-based array, and swaps each value with its index position.
 * In other words, for an output tensor y and an input tensor x, this operation computes the
 * following:
 *     y[x[i]] = i for i in [0, 1, ..., len(x) - 1]
 * The values must include 0. There can be no duplicate values or negative values.
 *
 * @tparam T Parameter numeric type. Only support float/double now
 */
private[bigdl] class InvertPermutation[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[Int], Tensor[Int], T] {

  output = Tensor[Int]()

  override def updateOutput(input: Tensor[Int]): Tensor[Int] = {
    require(input.dim() == 1, "InvertPermutation only accept 1D tensor as input")
    output.resizeAs(input)
    var i = 0
    while(i < input.size(1)) {
      output.setValue(input.valueAt(i + 1) + 1, i)
      i += 1
    }

    output
  }
}

private[bigdl] class ConcatOffset[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T] {

  output = T()

  override def updateOutput(input: Table): Table = {
    val concatDim = input[Tensor[Int]](1)
    require(concatDim.isScalar, "ConcatOffset: concat dim must be a scalar")
    val cdim = concatDim.value()
    val n = input.length() - 1
    var i = 1
    var offset = 0
    while(i <= n) {
      val shape = input[Tensor[Int]](i + 1)
      require(shape.nDimension() == 1, "ConcatOffset: shape must be 1D tensor")
      if (!output.contains(i)) {
        output(i) = Tensor[Int]()
      }
      val outputOffset = output[Tensor[Int]](i)
      outputOffset.resizeAs(shape).zero()
      outputOffset.setValue(cdim + 1, offset)
      val dimSize = shape.valueAt(cdim + 1)
      offset += dimSize
      i += 1
    }

    output
  }
}
