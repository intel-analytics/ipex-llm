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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, IdentityOutputShape}
import com.intel.analytics.bigdl.dllib.nn.keras.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * The local response normalization layer performs a kind of "lateral inhibition"
 * by normalizing over local input regions. The local regions extend spatially,
 * in separate channels (i.e., they have shape 1 x size x size).
 *
 * When you use this layer as the first layer of a model, you need to provide
 * the argument inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param size The side length of the square region to sum over. Default is 5.
 * @param alpha The scaling parameter. Default is 1.0.
 * @param beta The exponent. Default is 0.75.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class WithinChannelLRN2D[T: ClassTag](
    val size: Int = 5,
    val alpha: Double = 1.0,
    val beta: Double = 0.75,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"WithinChannelLRN2D requires 4D input, but got input dim ${input.length}")
    inputShape
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.dllib.nn.SpatialWithinChannelLRN(size, alpha, beta)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object WithinChannelLRN2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: Int = 5,
    alpha: Double = 1.0,
    beta: Double = 0.75,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): WithinChannelLRN2D[T] = {
    new WithinChannelLRN2D[T](size, alpha, beta, inputShape)
  }
}
