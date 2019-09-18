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

package com.intel.analytics.bigdl.nn.onnx

import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


/**
 * A layer which takes a tensor as input and outputs an 1D tensor containing the shape of the input.
 * @param `classTag$T`
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
class Shape[T: ClassTag](implicit ev: TensorNumeric[T])
  extends Operation[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dimSize = input.nDimension()
    output = Tensor[T](dimSize)
    (1 to dimSize).foreach(i => {
      output.setValue(i, ev.fromType(input.size(i)))
    })
    output
  }

}

object Shape {
  def apply[T: ClassTag]()(
    implicit ev: TensorNumeric[T]): Shape[T] = {
    new Shape[T]()
  }
}
