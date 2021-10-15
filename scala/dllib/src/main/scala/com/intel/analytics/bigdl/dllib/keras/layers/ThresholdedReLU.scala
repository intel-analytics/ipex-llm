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

import com.intel.analytics.bigdl.dllib.nn.keras.{ThresholdedReLU => BigDLThresholdedReLU}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Thresholded Rectified Linear Unit.
 * It follows:
 * f(x) = x for x > theta,
 * f(x) = 0 otherwise.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param theta Double >= 0. Threshold location of activation. Default is 1.0.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ThresholdedReLU[T: ClassTag](
    override val theta: Double = 1.0,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLThresholdedReLU[T] (
    theta, inputShape) with Net {
}

object ThresholdedReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
    theta: Double = 1.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ThresholdedReLU[T] = {
    new ThresholdedReLU[T](theta, inputShape)
  }
}
