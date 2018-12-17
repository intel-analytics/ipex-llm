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
import com.intel.analytics.bigdl.tensor.TensorNumericMath._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Returns (x - y)(x - y) element-wise.
 */
class SquaredDifference[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[_], T] {

  def updateOutput(inputs: Table): Tensor[_] = {
    val x = inputs[Tensor[NumericWildcard]](1)
    val y = inputs[Tensor[NumericWildcard]](2)

    require(x.getType() == y.getType(), "The numeric type of x and y must be the same, but got" +
      s"x: ${x.getType()}, y: ${y.getType()}")

    if (output.getType() != x.getType()) {
      output = x.emptyInstance()
    }

    output.asInstanceOf[Tensor[NumericWildcard]]
      .resizeAs(x).copy(x).sub(y).square()

    output
  }

}

object SquaredDifference {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): SquaredDifference[T]
  = new SquaredDifference()
}
