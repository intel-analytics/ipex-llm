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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{Shape, T, Table}
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.LayerWrapperByForward
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Creates a module that takes a table as input and outputs the element at index `index`
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape.
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param index the index to be selected. 0-based index.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SelectTable[T: ClassTag](index: Int, val inputShape: Shape = null)
  (implicit ev: TensorNumeric[T])
  extends LayerWrapperByForward[T](KerasUtils.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    val layer = com.intel.analytics.bigdl.dllib.nn.SelectTable(index + 1)
    layer.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object SelectTable {
  def apply[@specialized(Float, Double) T: ClassTag](index: Int, inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]) : SelectTable[T] = {
    new SelectTable[T](index, inputShape)
  }
}
