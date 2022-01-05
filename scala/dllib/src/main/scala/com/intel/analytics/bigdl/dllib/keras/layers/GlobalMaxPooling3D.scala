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

import com.intel.analytics.bigdl.dllib.nn.abstractnn._
import com.intel.analytics.bigdl.dllib.nn.VolumetricMaxPooling
import com.intel.analytics.bigdl.dllib.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.dllib.nn.internal.GlobalPooling3D
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies global max pooling operation for 3D data.
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * Border mode currently supported for this layer is 'valid'.
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dimOrdering Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GlobalMaxPooling3D[T: ClassTag](
   override val dimOrdering: String = "CHANNEL_FIRST",
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends GlobalPooling3D[T](dimOrdering, inputShape) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    val layer = VolumetricMaxPooling(
      kT = input(2),
      kW = input(4),
      kH = input(3),
      dT = 1,
      dW = 1,
      dH = 1)
    model.add(layer)
    model.add(com.intel.analytics.bigdl.dllib.nn.Squeeze(5))
    model.add(com.intel.analytics.bigdl.dllib.nn.Squeeze(4))
    model.add(com.intel.analytics.bigdl.dllib.nn.Squeeze(3))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object GlobalMaxPooling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : GlobalMaxPooling3D[T] = {
    new GlobalMaxPooling3D[T](KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
