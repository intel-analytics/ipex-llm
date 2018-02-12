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
 * S-shaped Rectified Linear Unit.
 * It follows:
 * f(x) = t^r + a^r(x - t^r) for x >= t^r,
 * f(x) = x for t^r > x > t^l,
 * f(x) = t^l + a^l(x - t^l) for x <= t^l.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param SharedAxes Array of Int. The axes along which to share learnable parameters
 *                   for the activation function. Default is null.
 *                   For example, if the incoming feature maps are from a 2D convolution
 *                   with output shape (batch, height, width, channels),
 *                   and you wish to share parameters across space
 *                   so that each filter only has one set of parameters,
 *                   set 'SharedAxes=Array(1,2)'.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SReLU[T: ClassTag](
   SharedAxes: Array[Int] = null,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val shape = inputShape.toSingle().toArray
    val layer = com.intel.analytics.bigdl.nn.SReLU(shape.slice(1, shape.length), SharedAxes)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
    SharedAxes: Array[Int] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SReLU[T] = {
    new SReLU[T](SharedAxes, inputShape)
  }
}
