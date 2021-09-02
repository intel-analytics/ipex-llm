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
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, Table}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalSplitTensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Creates a module that takes a Tensor as input and
 * outputs tables, splitting the Tensor along
 * the specified dimension `dimension`. Please note the dimension starts from 0.
 *
 * @param dimension to be split along this dimension
 * @param num elements number in the table
 * @tparam T Numeric type. Only support float/double now
 */
class SplitTensor[T: ClassTag](val dimension: Int, val num: Int,
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Table, T](KerasUtils.addBatch(inputShape)) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Table, T] = {
    val layer = new InternalSplitTensor(dimension + 1, num)
    layer.asInstanceOf[AbstractModule[Tensor[T], Table, T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val sizes = inputShape.toSingle()
    MultiShape(List.fill[Shape](num)(
      Shape(Array(sizes(0), sizes(dimension) / num) ++ sizes.drop(2))))
  }
}

object SplitTensor {
  def apply[@specialized(Float, Double) T: ClassTag](dimension: Int, num: Int,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SplitTensor[T] = {
    new SplitTensor[T](dimension, num, inputShape)
  }
}
