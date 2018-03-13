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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Abstract class for different pooling 1D layers.
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'AveragePooling1D' and 'MaxPooling1D' instead.
 */
abstract class Pooling1D[T: ClassTag](
   val poolLength: Int = 2,
   val stride: Int = -1,
   val borderMode: String = "valid",
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  // -1 means stride by default to be poolLength
  require(stride == -1 || stride > 0, s"Invalid stride value for Pooling1D: $stride")
  val strideValue: Int = if (stride > 0) stride else poolLength

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"Pooling1D requires 3D input, but got input dim ${input.length}")
    val outputLength = KerasUtils.computeConvOutputLength(input(1), poolLength,
      borderMode, strideValue)
    Shape(input(0), outputLength, input(2))
  }
}
