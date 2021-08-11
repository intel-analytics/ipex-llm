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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It is a simple layer which applies a mean operation over the given dimension.
 * When nInputDims is provided, the input will be considered as batches.
 * Then the mean operation will be applied in (dimension + 1).
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using `nInputDims`.
 *
 * @param dimension the dimension to be applied mean operation
 * @param nInputDims specify the number of dimensions that this module will receive
 *                   If it is more than the dimension of input tensors, the first dimension
 *                   would be considered as batch size
 * @param squeeze default is true, which will squeeze the sum dimension; set it to false to keep
 *                the sum dimension
 */

@SerialVersionUID(2995626598003841724L)
class Mean[T: ClassTag](
  val dimension: Int = 1,
  val nInputDims: Int = -1,
  val squeeze: Boolean = true)
  (implicit ev: TensorNumeric[T])
  extends Sum[T](dimension, nInputDims, true, squeeze) {
  override def toString: String = s"nn.Mean"
}

object Mean {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int = 1,
      nInputDims: Int = -1,
      squeeze: Boolean = true)(implicit ev: TensorNumeric[T]) : Mean[T] = {
    new Mean[T](dimension, nInputDims, squeeze)
  }
}
