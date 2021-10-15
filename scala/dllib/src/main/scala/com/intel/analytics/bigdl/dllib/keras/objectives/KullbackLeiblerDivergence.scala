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

import com.intel.analytics.bigdl.dllib.nn.KullbackLeiblerDivergenceCriterion
import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Loss calculated as:
 * y_true = K.clip(y_true, K.epsilon(), 1)
 * y_pred = K.clip(y_pred, K.epsilon(), 1)
 * and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)
 */
class KullbackLeiblerDivergence[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends TensorLossFunction[T]{

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    KullbackLeiblerDivergenceCriterion[T]()
}

object KullbackLeiblerDivergence {
  def apply[@specialized(Float, Double) T: ClassTag]()
   (implicit ev: TensorNumeric[T]): KullbackLeiblerDivergence[T] = {
    new KullbackLeiblerDivergence[T]()
  }
}
