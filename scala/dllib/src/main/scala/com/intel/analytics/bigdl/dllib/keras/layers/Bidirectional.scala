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

import com.intel.analytics.bigdl.dllib.nn.internal.{Bidirectional => BBidirectional}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Bidirectional wrapper for RNNs.
 * Bidirectional currently requires RNNs to return the full sequence, i.e. returnSequences = true.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * Example of creating a Bidirectional LSTM:
 * Bidirectional(LSTM(12, returnSequences = true), mergeMode = "sum", inputShape = Shape(32, 32))
 *
 * @param layer An instance of a recurrent layer.
 * @param mergeMode Mode by which outputs of the forward and backward RNNs will be combined.
 *                  Must be one of: 'sum', 'mul', 'concat', 'ave'. Default is 'concat'.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Bidirectional[T: ClassTag](
   override val layer: com.intel.analytics.bigdl.dllib.nn.internal.Recurrent[T],
   override val mergeMode: String = "concat",
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BBidirectional[T] (
    layer, mergeMode, inputShape) with Net {
}

object Bidirectional {
  def apply[@specialized(Float, Double) T: ClassTag](
    layer: com.intel.analytics.bigdl.dllib.nn.internal.Recurrent[T],
    mergeMode: String = "concat",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Bidirectional[T] = {
    new Bidirectional[T](layer, mergeMode, inputShape)
  }
}
