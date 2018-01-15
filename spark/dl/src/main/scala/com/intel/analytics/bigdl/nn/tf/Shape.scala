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
 * Given input, return the shape of this input as a 1-D tensor
 */
@SerialVersionUID(-907995771209831179L)
private[bigdl] class Shape[T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[Int], T] {

  override def updateOutput(input: Tensor[T]): Tensor[Int] = {
    this.output = Tensor[Int](input.size(), Array(input.nDimension()))
    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[Int]): Tensor[T] = {
    gradInput.resizeAs(input)
    gradInput.zero()
    gradInput
  }
}

private[bigdl] object Shape {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Shape[T] = {
    new Shape[T]()
  }
}


