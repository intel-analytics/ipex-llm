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

import com.intel.analytics.bigdl.nn.keras.{Recurrent => BigDLRecurrent}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

/**
 * This is the abstract base class for recurrent layers.
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'SimpleRNN', 'LSTM' and 'GRU' instead.
 */
abstract class Recurrent[T: ClassTag](
   override val outputDim: Int,
   override val returnSequences: Boolean = false,
   override val goBackwards: Boolean = false,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLRecurrent[T](
    outputDim, returnSequences, goBackwards, inputShape) with Net {
}
