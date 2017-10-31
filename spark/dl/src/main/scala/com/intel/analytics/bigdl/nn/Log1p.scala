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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The [[Log]] module applies a log transformation to the input data
 */
@SerialVersionUID(952324213749625368L)
class Log1p[T: ClassTag] (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  private val buffer: Tensor[T] = Tensor[T]()
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
      .copy(input)
      .log1p()
    output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    buffer.resizeAs(input)
    buffer.copy(input).add(ev.fromType[Double](1.0))
    gradInput.resizeAs(input)
      .fill(ev.fromType[Double](1.0))
      .cdiv(buffer)
      .cmul(gradOutput)

    gradInput
  }
}

object Log1p {
  def apply[@specialized(Float, Double) T: ClassTag]()
        (implicit ev: TensorNumeric[T]) : Log1p[T] = {
    new Log1p[T]()
  }
}
