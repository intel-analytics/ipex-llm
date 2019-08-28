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

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


/**
 * A layer contains a constant tensor.
 * It forwards the value of constant tensor to next layers.
 *
 * @param value constant tensor
 * @param `classTag$T`
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
class Constant[T: ClassTag](
  val value: Tensor[T]
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = value
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }

}

object Constant {
  def apply[@specialized(Float, Double) T: ClassTag](
        value: Tensor[T]
    ) (implicit ev: TensorNumeric[T]) : Constant[T] = {
    new Constant[T](value)
  }
}

