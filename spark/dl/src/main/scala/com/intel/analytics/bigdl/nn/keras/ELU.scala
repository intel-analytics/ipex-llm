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
  * Exponential Linear Unit.
  * It follows:
  * `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
  * `f(x) = x for x >= 0`.
  *
  * @param alpha scale for the negative factor.
  */
class ELU[T: ClassTag](
   val alpha: Double = 1.0,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.ELU(
      alpha = alpha,
      inplace = false)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ELU {
  def apply[@specialized(Float, Double) T: ClassTag](
    alpha: Double = 1.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : ELU[T] = {
    new ELU[T](alpha, inputShape)
  }
}
