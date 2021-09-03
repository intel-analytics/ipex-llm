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
import com.intel.analytics.bigdl.nn.keras.{Flatten => BigDLFlatten}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

/**
 * Flattens the input without affecting the batch size.
 * For example, if inputShape = Shape(2, 3, 4),
 * then outputShape will be Shape(24) with batch dimension unchanged.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Flatten[T: ClassTag](
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLFlatten[T](inputShape) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer =
      com.intel.analytics.bigdl.nn.Reshape(
        Array(input.slice(1, input.length).product), batchMode = Some(true))
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape) ++
      Net.param(getName())
    Net.kerasDef(this, params)
  }
}

object Flatten {
  def apply[@specialized(Float, Double) T: ClassTag](
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Flatten[T] = {
    new Flatten[T](inputShape)
  }
}

