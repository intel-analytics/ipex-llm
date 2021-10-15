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
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers.internal.InternalExpand
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

class Expand[T: ClassTag](tgtSizes: Array[Int],
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends LayerWrapperByForward[T](KerasUtils.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    InternalExpand[T](tgtSizes)
  }
}

object Expand {
  def apply[@specialized(Float, Double) T: ClassTag](tgtSizes: Array[Int],
    inputShape: Shape = null)(
    implicit ev: TensorNumeric[T]) : Expand[T] = {
    new Expand[T](tgtSizes, inputShape)
  }
}
