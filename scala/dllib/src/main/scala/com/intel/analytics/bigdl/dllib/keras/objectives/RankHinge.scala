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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.autograd.{CustomLoss, Variable, AutoGrad => A}

import scala.reflect.ClassTag

/**
 * Hinge loss for pairwise ranking problems.
 *
 * @param margin Double. Default is 1.0.
 */
class RankHinge[@specialized(Float, Double) T: ClassTag](
    margin: Double = 1.0)(implicit ev: TensorNumeric[T]) extends TensorLossFunction[T] {
  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    CustomLoss[T](RankHinge.marginLoss[T](margin), Shape(2, 1))
}

object RankHinge {
  def apply[@specialized(Float, Double) T: ClassTag](margin: Double = 1.0)
      (implicit ev: TensorNumeric[T]): RankHinge[T] = {
    new RankHinge[T](margin)
  }

  def marginLoss[T: ClassTag](margin: Double = 1.0)(implicit ev: TensorNumeric[T]):
  (Variable[T], Variable[T]) => Variable[T] = {

    def rankHingeLoss(yTrue: Variable[T], yPred: Variable[T])
     (implicit ev: TensorNumeric[T]): Variable[T] = {
      val target = yTrue - yTrue + yPred
      val pos = target.indexSelect(1, 0)
      val neg = target.indexSelect(1, 1)
      A.maximum(neg - pos + margin, 0)
    }
    rankHingeLoss
  }
}
