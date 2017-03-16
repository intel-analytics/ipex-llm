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

@SerialVersionUID(1208478077576570643L)
class ReLU[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends Threshold[T](0, 0, ip) {

  override def toString(): String = {
    s"nn.ReLU"
  }
}

object ReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
      ip: Boolean = false)(implicit ev: TensorNumeric[T]) : ReLU[T] = {
    new ReLU[T](ip)
  }
}
