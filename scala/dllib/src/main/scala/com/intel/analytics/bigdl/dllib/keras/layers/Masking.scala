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

import com.intel.analytics.bigdl.dllib.nn.keras.{Masking => BigDLMasking}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Use a mask value to skip timesteps for a sequence.
 * Masks a sequence by using a mask value to skip timesteps.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param maskValue Double, mask value.
 *                  For each timestep in the input (the second dimension),
 *                  if all the values in the input at that timestep are equal to 'maskValue',
 *                  then the timestep will masked (skipped) in all downstream layers.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Masking[T: ClassTag](
    override val maskValue: Double = 0.0,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLMasking[T](
    maskValue, inputShape) with Net {
}

object Masking {
  def apply[@specialized(Float, Double) T: ClassTag](
    maskValue: Double = 0.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Masking[T] = {
    new Masking[T](maskValue, inputShape)
  }
}
