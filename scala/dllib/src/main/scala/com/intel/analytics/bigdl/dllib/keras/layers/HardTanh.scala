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
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies the hard tanh function element-wise to the input.
 *
 *          ⎧  maxValue, if x > maxValue
 *   f(x) = ⎨  minValue, if x < minValue
 *          ⎩  x, otherwise
 *
 * When you use this layer as the first layer of a model, you need to provide
 * the argument inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param minValue The minimum threshold value. Default is -1.
 * @param maxValue The maximum threshold value. Default is 1.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class HardTanh[T: ClassTag](
    val minValue: Double = -1,
    val maxValue: Double = 1,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with IdentityOutputShape with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.dllib.nn.HardTanh(minValue, maxValue)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object HardTanh {
  def apply[@specialized(Float, Double) T: ClassTag](
    minValue: Double = -1,
    maxValue: Double = 1,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): HardTanh[T] = {
    new HardTanh[T](minValue, maxValue, inputShape)
  }
}
