/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Narrow the input with the number of dimensions not being reduced.
 * The batch dimension needs to be unchanged.
 * For example, if input is:
 * 1 2 3
 * 4 5 6
 * Narrow(1, 1, 2) will give output
 * 2 3
 * 5 6
 * Narrow(1, 2, -1) will give output
 * 3
 * 6
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param dim The dimension to narrow. 0-based index. Cannot narrow the batch dimension.
 *            -1 means the last dimension of the input.
 * @param offset Non-negative integer. The start index on the given dimension. 0-based index.
 * @param length The length to narrow. Default is 1.
 *               Can use a negative length such as -1 in the case where input size is unknown.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Narrow[T: ClassTag](
    val dim: Int,
    val offset: Int,
    val length: Int = 1,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  private def getPositiveDimAndLength(inputShape: Shape): (Int, Int) = {
    val input = inputShape.toSingle().toArray
    val positiveDim = if (dim < 0) dim + input.length else dim
    require(positiveDim >= 0 && positiveDim <= input.length - 1,
      s"Invalid select dim: $dim, dim should be within range [0, ${input.length - 1}]")
    val positiveLength = if (length < 0) length + input(positiveDim) - offset + 1 else length
    // batch dimension is always -1 for now, so we skip the checking here.
    if(dim > 0) {
      require(offset >= 0 && offset <= input(positiveDim) -1,
        s"Invalid narrow offset for dim $dim: $offset, " +
          s"offset should be within range [0, ${input(positiveDim) - 1}]")
      require(positiveLength > 0 && positiveLength <= input(positiveDim) - offset,
        s"Invalid narrow length for dim $dim with offset $offset: $length, " +
          s"length should be within range (0, ${input(positiveDim) - offset}]")
    }
    (positiveDim, positiveLength)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val (positiveDim, positiveLength) = getPositiveDimAndLength(inputShape)
    inputShape.copyAndUpdate(positiveDim, positiveLength)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val (positiveDim, positiveLength) = getPositiveDimAndLength(inputShape)
    val layer = com.intel.analytics.bigdl.nn.Narrow(positiveDim + 1, offset + 1, positiveLength)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Narrow {
  def apply[@specialized(Float, Double) T: ClassTag](
    dim: Int,
    offset: Int,
    length: Int = 1,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Narrow[T] = {
    new Narrow[T](dim, offset, length, inputShape)
  }
}
