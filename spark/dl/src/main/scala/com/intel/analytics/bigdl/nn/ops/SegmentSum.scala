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

/**
 * Computes the sum along segments of a tensor.
 */
class SegmentSum[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[T], T]{

  def updateOutput(inputs: Table): Tensor[T] = {
    val x = inputs[Tensor[T]](1)
    val y = inputs[Tensor[Int]](2) // zero-indices
    require(y.nDimension() == 1, "segment ids should be 1D tensor")
    require(y.size(1) == x.size(1), "segment ids should be the same size as" +
      s" first dimension of input, excepted ${x.size(1)}, but got ${y.size(1)}")
    val newSize = x.size()
    newSize(0) = y.valueAt(y.nElement()) + 1
    output.resize(newSize).zero()

    var i = 0
    while(i < y.nElement()) {
      output.select(1, y.valueAt(i + 1) + 1).add(x.select(1, i + 1))
      i += 1
    }

    output
  }

}

object SegmentSum {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): SegmentSum[T] = {
    new SegmentSum()
  }
}
