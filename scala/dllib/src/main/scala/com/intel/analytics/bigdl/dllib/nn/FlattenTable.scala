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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * This is a table layer which takes an arbitrarily deep table of Tensors
 * (potentially nested) as input and a table of Tensors without any nested
 * table will be produced
 */

@SerialVersionUID(7620301574431959449L)
class FlattenTable[T: ClassTag] (implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {

  override def updateOutput(input: Table): Table = {
    output = input.flatten()
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = gradOutput.inverseFlatten(input)
    gradInput
  }

  override def toString: String = {
    s"nn.Flatten"
  }
}

object FlattenTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : FlattenTable[T] = {
    new FlattenTable[T]()
  }
}
