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

package com.intel.analytics.bigdl.dllib.keras.layers.internal

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.nn.JoinTable
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{T, Table}

import scala.reflect.ClassTag

class InternalSplitTensor[T: ClassTag](dimension: Int, num: Int)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Table, T] {

  override def updateOutput(input: Tensor[T]): Table = {
    output =
      T.array(input.split(input.size(dimension) / num, dimension))
    output
  }

  private val innerLayer = new JoinTable[T](dimension, -1)
  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    gradInput = innerLayer.forward(gradOutput).toTensor[T]
    gradInput
  }
}
