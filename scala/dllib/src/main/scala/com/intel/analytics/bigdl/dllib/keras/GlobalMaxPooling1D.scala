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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Applies global max pooling operation for temporal data.
 * The input of this layer should be 3D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GlobalMaxPooling1D[T: ClassTag](
   inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends GlobalPooling1D[T](inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    model.add(com.intel.analytics.bigdl.nn.Reshape(Array(input(1), 1, input(2)), Some(true)))
    val layer = SpatialMaxPooling(
      kW = 1,
      kH = input(1),
      format = DataFormat.NHWC)
    model.add(layer)
    model.add(Squeeze(3))
    model.add(Squeeze(2))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object GlobalMaxPooling1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : GlobalMaxPooling1D[T] = {
    new GlobalMaxPooling1D[T](inputShape)
  }
}
