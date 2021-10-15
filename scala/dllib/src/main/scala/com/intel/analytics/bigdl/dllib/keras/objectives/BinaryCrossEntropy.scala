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
import com.intel.analytics.bigdl.dllib.nn.BCECriterion
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.tensor.Tensor

/**
 * This loss function measures the Binary Cross Entropy between the target and the output
 *         loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
 * or in the case of the weights argument being specified:
 *         loss(o, t) = - 1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
 *
 * @param weights weights over the input dimension
 * @param sizeAverage Boolean. Whether losses are averaged over observations for each
                      mini-batch. Default is true. If false, the losses are instead
                      summed for each mini-batch.
 */
class BinaryCrossEntropy[@specialized(Float, Double) T: ClassTag](
  val weights: Tensor[T] = null, sizeAverage: Boolean = true)
(implicit ev: TensorNumeric[T]) extends TensorLossFunction[T]{

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    BCECriterion[T](weights, sizeAverage)
}

object BinaryCrossEntropy {
  def apply[@specialized(Float, Double) T: ClassTag](
      weights: Tensor[T] = null,
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]): BinaryCrossEntropy[T] = {
    new BinaryCrossEntropy[T](weights, sizeAverage)
  }
}
