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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The [[Log]] module applies a log transformation to the input data
 */
@SerialVersionUID(952324213749625368L)
class Log1p[T: ClassTag, D: ClassTag] (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T] {

  output = Tensor[D]()
  gradInput = Tensor[D]()

  private val buffer: Tensor[D] = Tensor[D]()
  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    output.resizeAs(input)
      .copy(input)
      .log1p()
    output
  }
  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    buffer.resizeAs(input)
    buffer.copy(input).add(ev2.fromType[Double](1.0))
    gradInput.resizeAs(input)
      .fill(ev2.fromType[Double](1.0))
      .cdiv(buffer)
      .cmul(gradOutput)

    gradInput
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object Log1p {
  def apply[T: ClassTag, D: ClassTag]()
        (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) : Log1p[T, D] = {
    new Log1p[T, D]()
  }
}
