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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
  * Parametric Softplus.
  * It follows:
  * `f(x) = alpha * log(1 + exp(beta * x))`
  * Only alpha_init = 1/beta_init is supported for now.
  *
  * @param beta float. Initial values of the beta weights.
  */
class ParametricSoftPlus[T: ClassTag](
   val beta: Double = 1.0,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
      val layer = SoftPlus(beta = beta)
      layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ParametricSoftPlus {
  def apply[@specialized(Float, Double) T: ClassTag](
    beta: Double = 1.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : ParametricSoftPlus[T] = {
    new ParametricSoftPlus[T](beta, inputShape)
  }
}
