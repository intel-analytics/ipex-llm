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

import com.intel.analytics.bigdl.nn.abstractnn.{TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc2, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This class is same as `Poisson` loss in keras.
 * @param ev
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class PoissonCriterion[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private val epsilon: T = ev.fromType(1e-07)

  // K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    val buffer = Tensor[T]()
    buffer.resizeAs(target).copy(target)

    buffer.apply1(e => ev.plus(e, epsilon))
    buffer.log().cmul(target)
    buffer.negative(buffer).add(input).mean()
  }

  // 1 - y_true/y_pred
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      "Input should have the same size as gradOutput" +
        s"input size(${input.dim()}) gradOutput size(${gradOutput.dim()})")
    gradInput.resizeAs(input).copy(input)

    gradInput.div(gradOutput)
    gradInput.negative(gradInput).add(ev.fromType[Double](1.0))

    gradInput
  }
}

object PoissonCriterion {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): PoissonCriterion[T] =
    new PoissonCriterion[T]()
}