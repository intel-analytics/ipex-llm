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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Apply multiplicative 1-centered Gaussian noise.
 * As it is a regularization layer, it is only active at training time.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param p Double, drop probability (as with 'Dropout').
 *          The multiplicative noise will have standard deviation 'sqrt(p/(1-p))'.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GaussianDropout[T: ClassTag](
   val p: Double,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape))
    with IdentityOutputShape {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.GaussianDropout(rate = p)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object GaussianDropout {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): GaussianDropout[T] = {
    new GaussianDropout[T](p, inputShape)
  }
}
