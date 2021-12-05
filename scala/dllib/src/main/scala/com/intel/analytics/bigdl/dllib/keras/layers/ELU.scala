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

import com.intel.analytics.bigdl.dllib.nn.internal.{ELU => BigDLELU}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Exponential Linear Unit.
 * It follows:
 * f(x) =  alpha * (exp(x) - 1.) for x < 0,
 * f(x) = x for x >= 0.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param alpha Double, scale for the negative factor. Default is 1.0.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ELU[T: ClassTag](
    override val alpha: Double = 1.0,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLELU[T](
    alpha, inputShape) with Net {
}

object ELU {
  def apply[@specialized(Float, Double) T: ClassTag](
    alpha: Double = 1.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ELU[T] = {
    new ELU[T](alpha, inputShape)
  }
}
