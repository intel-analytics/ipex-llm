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
 * Applies the randomized leaky rectified linear unit element-wise to the input.
 *
 * f(x) = max(0,x) + a * min(0, x) where a ~ U(l, u).
 *
 * In the training mode, negative inputs are multiplied by a factor drawn
 * from a uniform random distribution U(l, u).
 * In the evaluation mode, a RReLU behaves like a LeakyReLU with a constant mean
 * factor a = (l + u) / 2.
 * If l == u, a RReLU essentially becomes a LeakyReLU.
 * Regardless of operating in in-place mode a RReLU will internally
 * allocate an input-sized noise tensor to store random factors for negative inputs.
 * For reference, see [Empirical Evaluation of Rectified Activations in Convolutional
 * Network](http://arxiv.org/abs/1505.00853).
 *
 * When you use this layer as the first layer of a model, you need to provide
 * the argument inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param lower Lower boundary of the uniform random distribution. Default is 1.0/8.
 * @param upper Upper boundary of the uniform random distribution. Default is 1.0/3.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class RReLU[T: ClassTag](
    val lower: Double = 1.0/8,
    val upper: Double = 1.0/3,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with IdentityOutputShape with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.dllib.nn.RReLU(lower, upper)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object RReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
    lower: Double = 1.0/8,
    upper: Double = 1.0/3,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): RReLU[T] = {
    new RReLU[T](lower, upper, inputShape)
  }
}
