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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

/**
 * Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each
 * update during training time in order to prevent overfitting.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param p Fraction of the input units to drop. Double between 0 and 1.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Dropout[T: ClassTag](
  override val p: Double,
  override val inputShape: Shape = null)
  (implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.nn.keras.Dropout[T](p, inputShape) with Net {

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.param(getName()) ++
      Net.param(p, "rate")
    Net.kerasDef(this, params)
  }

}

object Dropout {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double,
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): Dropout[T] = {
    new Dropout[T](p, inputShape)
  }
}
