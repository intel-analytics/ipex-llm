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

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn.Threshold
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

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
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ThresholdedReLU[T: ClassTag](
   val theta: Double = 1.0,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape))
    with IdentityOutputShape {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = Threshold(
      th = theta,
      v = 0.0,
      ip = false)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ThresholdedReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
    theta: Double = 1.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ThresholdedReLU[T] = {
    new ThresholdedReLU[T](theta, inputShape)
  }
}
