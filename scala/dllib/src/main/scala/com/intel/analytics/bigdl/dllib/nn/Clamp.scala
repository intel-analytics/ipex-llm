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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * A kind of hard tanh activition function with integer min and max
 * @param minV min value
 * @param maxV max value
 * @param ev numeric operator
 * @tparam T numeric type
 */
@SerialVersionUID(- 3787689437971361185L)
class Clamp[T: ClassTag](private val minV: Int, private val maxV: Int)(
  implicit ev: TensorNumeric[T]) extends HardTanh[T](minV, maxV) {
  override def toString(): String = {
    s"nn.Clamp"
  }
}

object Clamp {
  def apply[@specialized(Float, Double) T: ClassTag](
      min: Int,
      max: Int)(implicit ev: TensorNumeric[T]) : Clamp[T] = {
    new Clamp[T](min, max)
  }
}
