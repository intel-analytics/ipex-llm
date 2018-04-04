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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, Input => BInput, InputLayer => BInputLayer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Used to instantiate an input node.
 */
object Input {
  def apply[T: ClassTag](
      inputShape: Shape = null,
      name : String = null)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    BInput(inputShape, name)
  }
}

/**
 * Used as an entry point into a model.
 */
object InputLayer {
  def apply[T: ClassTag](
      inputShape: Shape = null,
      name : String = null)(implicit ev: TensorNumeric[T]): KerasLayer[Activity, Activity, T] = {
    BInputLayer(inputShape, name)
  }
}
