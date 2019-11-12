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

package com.intel.analytics.zoo.pipeline.api.keras.objectives

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This is same with cross entropy criterion
 * except the target tensor is a one-hot tensor
 */
class CategoricalCrossEntropy[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends TensorLossFunction[T] {

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    com.intel.analytics.bigdl.nn.CategoricalCrossEntropy[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    val eps = ev.fromType(1e-8)
    val maxFloat = ev.fromType(Float.MaxValue)
    // avoid NaN when compute input's log in BigDL's CategoricalCrossEntropy
    input.apply1(ev.clip(_, eps, maxFloat))
    output = loss.updateOutput(input, target)
    output
  }
}

object CategoricalCrossEntropy {
  def apply[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]): CategoricalCrossEntropy[T] = {
    new CategoricalCrossEntropy[T]()
  }
}
