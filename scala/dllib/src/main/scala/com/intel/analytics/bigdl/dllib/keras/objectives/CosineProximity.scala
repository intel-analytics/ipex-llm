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

import com.intel.analytics.bigdl.dllib.nn.CosineProximityCriterion
import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * The negative of the mean cosine proximity between predictions and targets.
 * The cosine proximity is defined as below:
 * x'(i) = x(i) / sqrt(max(sum(x(i)^2), 1e-12))
 * y'(i) = y(i) / sqrt(max(sum(x(i)^2), 1e-12))
 * cosine_proximity(x, y) = mean(-1 * x'(i) * y'(i))
 */
class CosineProximity[@specialized(Float, Double) T: ClassTag]()
   (implicit ev: TensorNumeric[T]) extends TensorLossFunction[T]{

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    CosineProximityCriterion[T]()
}

object CosineProximity {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]): CosineProximity[T] = {
    new CosineProximity[T]()
  }
}
