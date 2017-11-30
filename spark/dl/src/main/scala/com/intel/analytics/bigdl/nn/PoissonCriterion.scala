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
 * This class is same as `Poisson` loss in keras.
 * Loss calculated as:
 * K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class PoissonCriterion[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private val epsilon: T = ev.fromType(1e-07)

  /*
   * K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
   */
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.isSameSizeAs(target),
      s"Input should have the same size as target. input size: (${input.size().mkString(", ")});" +
        s" target size: (${target.size().mkString(", ")}).")
    // use gradInput as buffer
    gradInput.resizeAs(input).copy(input)
    gradInput.add(epsilon).log().cmul(target).negative(gradInput).add(input).mean()
  }

  /*
   * back propagation with: 1 - y_true/y_pred
   */
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput),
      s"Input should have the same size as target. input size: (${input.size().mkString(", ")});" +
        s" target size: (${gradOutput.size().mkString(", ")}).")

    gradInput.resizeAs(gradOutput).copy(gradOutput)
    gradInput.div(input).negative(gradInput).add(ev.fromType[Double](1.0))
      .div(ev.fromType[Int](input.nElement()))
  }
}

object PoissonCriterion {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): PoissonCriterion[T] =
    new PoissonCriterion[T]()
}
