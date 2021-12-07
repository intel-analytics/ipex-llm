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

import com.intel.analytics.bigdl.dllib.nn.internal.{RepeatVector => BigDLRepeatVector}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Repeats the input n times.
 * The input of this layer should be 2D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param n Repetition factor. Integer.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class RepeatVector[T: ClassTag](
    override val n: Int,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLRepeatVector[T] (
    n, inputShape) with Net {
}

object RepeatVector {
  def apply[@specialized(Float, Double) T: ClassTag](
    n: Int,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): RepeatVector[T] = {
    new RepeatVector[T](n, inputShape)
  }
}
