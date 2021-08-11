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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Compute the dot product of input and target tensor.
 * Input and target are required to have the same size.
 * @param sizeAverage whether to average over each observations in the same batch
 */
@SerialVersionUID(3360838286914764710L)
class DotProductCriterion[T: ClassTag]
(sizeAverage: Boolean = false)(
         implicit ev: TensorNumeric[T]) extends TensorCriterion[T]  {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.dim() == 1 || input.dim() == 2, "DotProductCriterion only" +
      "support tensor with 1 or 2 dimensions")
    require(input.size().sameElements(target.size()), "The shape of input and target" +
      "must be the same")

    val dotProduct = target.dot(input)
    if (sizeAverage && input.dim() == 2) {
      output = ev.divide(dotProduct, ev.fromType(input.size(1)))
    } else {
      output = dotProduct
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "DotProductCriterion only" +
      "support tensor with 1 or 2 dimensions")
    require(input.size().sameElements(target.size()), "The shape of input and target" +
      "must be the same")

    gradInput.resizeAs(target)
    Tensor.dense(target, gradInput)
    if (sizeAverage && target.dim() == 2) {
      gradInput.div(ev.fromType(target.size(1)))
    }
    gradInput
  }
}

object DotProductCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = false)
      (implicit ev: TensorNumeric[T]) : DotProductCriterion[T] = {
    new DotProductCriterion[T]()
  }
}
