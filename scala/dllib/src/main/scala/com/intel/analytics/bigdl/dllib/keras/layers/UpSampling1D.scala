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

import com.intel.analytics.bigdl.dllib.nn.internal.{UpSampling1D => BigDLUpSampling1D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * UpSampling layer for 1D inputs.
 * Repeats each temporal step 'length' times along the time axis.
 * The input of this layer should be 3D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param length Integer. UpSampling factor. Default is 2.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class UpSampling1D[T: ClassTag](
   override val length: Int = 2,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLUpSampling1D[T](length, inputShape) with Net {}

object UpSampling1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    length: Int = 2,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): UpSampling1D[T] = {
    new UpSampling1D[T](length, inputShape)
  }
}
