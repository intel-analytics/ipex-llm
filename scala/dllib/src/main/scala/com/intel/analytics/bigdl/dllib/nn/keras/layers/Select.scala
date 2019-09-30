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
 * Select an index of the input in the given dim and return the subset part.
 * The batch dimension needs to be unchanged.
 * For example, if input is:
 * 1 2 3
 * 4 5 6
 * Select(1, 1) will give output [2 5]
 * Select(1, -1) will give output [3 6]
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param dim The dimension to select. 0-based index. Cannot select the batch dimension.
 *            -1 means the last dimension of the input.
 * @param index The index of the dimension to be selected. 0-based index.
 *              -1 means the last dimension of the input.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Select[T: ClassTag](
    val dim: Int,
    val index: Int,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  private def getPositiveDimAndIndex(inputShape: Shape): (Int, Int) = {
    val input = inputShape.toSingle().toArray
    require(input.length >= 2, s"Select requires >= 2D input, but got input dim ${input.length}")
    val positiveDim = if (dim < 0) dim + input.length else dim
    require(positiveDim != 0, "Cannot select the batch dimension")
    require(positiveDim > 0 && positiveDim <= input.length - 1,
      s"Invalid select dim: $dim, dim should be within range (0, ${input.length - 1}]")
    val positiveIndex = if (index < 0) index + input(positiveDim) else index
    require(positiveIndex >= 0 && positiveIndex <= input(positiveDim) - 1,
      s"Invalid select index for dim $dim: $index, " +
        s"index should be within range [0, ${input(positiveDim) - 1}]")
    (positiveDim, positiveIndex)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray.toBuffer
    input.remove(getPositiveDimAndIndex(inputShape)._1)
    Shape(input.toArray)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val (positiveDim, positiveIndex) = getPositiveDimAndIndex(inputShape)
    val layer = com.intel.analytics.bigdl.nn.Select(positiveDim + 1, positiveIndex + 1)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.param(getName()) ++
      Net.param(dim, "dim")
    Net.kerasDef(this, params)
  }
}

object Select {
  def apply[@specialized(Float, Double) T: ClassTag](
    dim: Int,
    index: Int,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Select[T] = {
    new Select[T](dim, index, inputShape)
  }
}
