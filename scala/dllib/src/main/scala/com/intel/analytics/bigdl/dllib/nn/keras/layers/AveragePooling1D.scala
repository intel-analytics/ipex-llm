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

import com.intel.analytics.bigdl.nn.keras.{AveragePooling1D => BigDLAveragePooling1D}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

/**
 * Applies average pooling operation for temporal data.
 * The input of this layer should be 3D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param poolLength Size of the region to which average pooling is applied. Integer. Default is 2.
 * @param stride Factor by which to downscale. Positive integer, or -1. 2 will halve the input.
 *               If -1, it will default to poolLength. Default is -1, and in this case it will
 *               be equal to poolSize.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class AveragePooling1D[T: ClassTag](
    poolLength: Int = 2,
    stride: Int = -1,
    borderMode: String = "valid",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLAveragePooling1D[T](
    poolLength, stride, borderMode, inputShape) with Net {
}

object AveragePooling1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolLength: Int = 2,
    stride: Int = -1,
    borderMode: String = "valid",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): AveragePooling1D[T] = {
    new AveragePooling1D[T](poolLength, stride, borderMode, inputShape)
  }
}
