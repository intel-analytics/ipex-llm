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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{Shape, Table}
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalMax
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies a max operation over dimension `dim`
 *
 * @param dim max along this dimension
 * @param numInputDims Optional. If in a batch model, set to the inputDims.
 * @param returnValue Optional. Config whether return value or indices
 *
 * @tparam T Numeric type. Only support float/double now
 */
class Max[T: ClassTag](dim: Int, numInputDims: Int = Int.MinValue, returnValue: Boolean = true,
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Table, T](KerasUtils.addBatch(inputShape)) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Table, T] = {
    val layer = new InternalMax[T](dim + 1, numInputDims, returnValue)
    layer.asInstanceOf[AbstractModule[Tensor[T], Table, T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val sizes = inputShape.toSingle().toArray
    if (sizes.length > 1) {
      Shape(sizes.updated(dim, 1))
    } else Shape(Array(1))
  }
}

object Max {
  def apply[@specialized(Float, Double) T: ClassTag](dim: Int,
    numInputDims: Int = Int.MinValue, returnValue: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Max[T] = {
    new Max[T](dim, numInputDims, returnValue, inputShape)
  }
}

