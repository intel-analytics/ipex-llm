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

package com.intel.analytics.bigdl.dllib.keras.objectives

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The base class for Keras-style API objectives in Analytics Zoo.
 *
 * @tparam A Input data type.
 * @tparam B Target data type.
 */
abstract class LossFunction[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractCriterion[A, B, T] {

  protected val loss: AbstractCriterion[A, B, T]

  override def updateOutput(input: A, target: B): T = {
    output = loss.updateOutput(input, target)
    output
  }

  def updateGradInput(input: A, target: B): A = {
    gradInput = loss.updateGradInput(input, target)
    gradInput
  }

}

/**
 * A subclass of LossFunction where input and target are both Tensors.
 */
abstract class TensorLossFunction[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends LossFunction[Tensor[T], Tensor[T], T]
