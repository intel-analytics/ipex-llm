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
 * Use a mask value to skip timesteps for a sequence.
 * Masks a sequence by using a mask value to skip timesteps.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param maskValue Double, mask value.
 *                  For each timestep in the input (the second dimension),
 *                  if all the values in the input at that timestep are equal to 'maskValue',
 *                  then the timestep will masked (skipped) in all downstream layers.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Masking[T: ClassTag](
   val maskValue: Double = 0.0,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape))
    with IdentityOutputShape {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.Masking(maskValue = maskValue)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Masking {
  def apply[@specialized(Float, Double) T: ClassTag](
    maskValue: Double = 0.0,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Masking[T] = {
    new Masking[T](maskValue, inputShape)
  }
}
