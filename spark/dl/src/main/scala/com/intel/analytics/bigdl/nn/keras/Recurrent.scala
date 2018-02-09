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

import com.intel.analytics.bigdl.nn.{Cell, Reverse, Select, Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * This is the abstract base class for recurrent layers.
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'SimpleRNN', 'LSTM' and 'GRU' instead.
 */
abstract class Recurrent[T: ClassTag](
   val outputDim: Int,
   val returnSequences: Boolean = false,
   val goBackwards: Boolean = false,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"Recurrent layers require 3D input, but got input dim ${input.length}")
    if (returnSequences) Shape(input(0), input(1), outputDim)
    else Shape(input(0), outputDim)
  }

  def buildCell(input: Array[Int]): Cell[T] = {
    throw new RuntimeException("Recurrent cell haven't been implemented yet.")
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    if (goBackwards) model.add(Reverse(2))
    val rec = com.intel.analytics.bigdl.nn.Recurrent[T]()
    rec.add(buildCell(input))
    model.add(rec)
    if (!returnSequences) model.add(Select(2, -1))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}
