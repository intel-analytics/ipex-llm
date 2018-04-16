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

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net

/**
 * Just a wrapper class. Please use Activation('softmax') instead.
 */
class SoftMax[T: ClassTag](
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.nn.keras.SoftMax[T](inputShape) with Net {
}

object SoftMax {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T](inputShape)
  }
}
