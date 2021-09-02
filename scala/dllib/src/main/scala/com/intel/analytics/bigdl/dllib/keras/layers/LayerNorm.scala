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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalLayerNorm
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Normalization layer used in Bert.
 * u = AutoGrad.mean(x, sizes.size - 1, true)
 * t = x - u
 * s = AutoGrad.mean(AutoGrad.square(x - u), sizes.size -1, true)
 * y = (x - u) / AutoGrad.sqrt(s + e)
 * y * weight + bias
 *
 * @param nOutput The size of output dimension.
 * @param eps Optional. Small value to avoid divide zero.
 *
 * @tparam T Numeric type. Only support float/double now
 */
class LayerNorm[T: ClassTag](val nOutput: Int = 768, val eps: Double = 1e-5,
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net{

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = new InternalLayerNorm[T](nOutput, eps)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    Shape(input.slice(0, input.length -1) ++ Array(nOutput))
  }
}

object LayerNorm {
  def apply[@specialized(Float, Double) T: ClassTag](nOutput: Int = 768,
    eps: Double = 1e-5,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LayerNorm[T] = {
    new LayerNorm[T](nOutput, eps, inputShape)
  }
}
