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

import com.intel.analytics.bigdl.nn.keras.{Permute => BigDLPermute}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

/**
 * Permutes the dimensions of the input according to a given pattern.
 * Useful for connecting RNNs and convnets together.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dims Int array. Permutation pattern, does not include the batch dimension.
 *             Indexing starts at 1.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Permute[T: ClassTag](
    override val dims: Array[Int],
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLPermute[T](
    dims, inputShape) with Net {

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.param(getName()) ++
      Net.arrayToString(dims, "dims")
    Net.kerasDef(this, params)
  }
}

object Permute {
  def apply[@specialized(Float, Double) T: ClassTag](
    dims: Array[Int],
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Permute[T] = {
    new Permute[T](dims, inputShape)
  }
}
