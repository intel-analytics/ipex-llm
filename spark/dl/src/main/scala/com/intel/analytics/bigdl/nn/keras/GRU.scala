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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class GRU[T: ClassTag](
   outputDim: Int,
   val activation: TensorModule[T] = null,
   val innerActivation: TensorModule[T] = null,
   returnSequences: Boolean = false,
   goBackwards: Boolean = false,
   var wRegularizer: Regularizer[T] = null,
   var uRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Recurrent[T](outputDim, returnSequences, goBackwards, inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = com.intel.analytics.bigdl.nn.GRU[T](
      inputSize = input(2),
      outputSize = outputDim,
      activation = activation,
      innerActivation = innerActivation,
      wRegularizer = wRegularizer,
      uRegularizer = uRegularizer,
      bRegularizer = bRegularizer)
    super.processParameters(layer)
  }
}

object GRU {
  def apply[@specialized(Float, Double) T: ClassTag](
    outputDim: Int,
    activation: String = "tanh",
    innerActivation: String = "hard_sigmoid",
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : GRU[T] = {
    new GRU(outputDim, KerasUtils.getActivation(activation),
      KerasUtils.getActivation(innerActivation), returnSequences,
      goBackwards, wRegularizer, uRegularizer, bRegularizer, inputShape)
  }
}
