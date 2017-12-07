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

import com.intel.analytics.bigdl.tensor.{IntType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag


class ArgMax[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[Int], T] {

  output = Tensor[Int]()

  override def updateOutput(input: Table): Tensor[Int] = {

    val inputTensor = input[Tensor[_]](1)
    val dimension = input[Tensor[Int]](2).value() + 1

    val (_, result) = inputTensor
      .asInstanceOf[Tensor[NumericWildcard]]
      .max(dimension)

    output.resizeAs(result)
    result.cast[Int](output)
    output.squeeze(dimension)

    output
  }

}

object ArgMax {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): ArgMax[T] = new ArgMax[T]()
}
