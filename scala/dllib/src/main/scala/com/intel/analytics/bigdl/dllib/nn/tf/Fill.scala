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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Creates a tensor filled with a scalar value. Input should be a 1-D tensor defining
 * the shape of the output tensor.
 * @param value the scalar value to be filled.
 */
@SerialVersionUID(-471757174144422555L)
private[bigdl] class Fill[T: ClassTag](value: T) (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (output.dim() == 0) {
      val shape = input.storage().array().map(ev.toType[Int])
      output = Tensor(shape).fill(value)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
  }

}

private[bigdl] object Fill {
  def apply[T: ClassTag](value: Double)
       (implicit ev: TensorNumeric[T]) : Fill[T] = {
    new Fill[T](ev.fromType(value))
  }
}
