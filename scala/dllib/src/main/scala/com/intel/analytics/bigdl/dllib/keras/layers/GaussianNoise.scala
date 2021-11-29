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

import com.intel.analytics.bigdl.dllib.nn.keras.{GaussianNoise => BigDLGaussianNoise}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Apply additive zero-centered Gaussian noise.
 * This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
 * Gaussian Noise is a natural choice as corruption process for real valued inputs.
 * As it is a regularization layer, it is only active at training time.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param sigma Double, standard deviation of the noise distribution.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GaussianNoise[T: ClassTag](
    override val sigma: Double,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLGaussianNoise[T](
    sigma, inputShape) with Net {
}

object GaussianNoise {
  def apply[@specialized(Float, Double) T: ClassTag](
    sigma: Double,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): GaussianNoise[T] = {
    new GaussianNoise[T](sigma, inputShape)
  }
}
