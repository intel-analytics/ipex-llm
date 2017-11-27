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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc2}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This method is same as `kullback_leibler_divergence` loss in keras.
 * It calculates:
 * y_true = K.clip(y_true, K.epsilon(), 1)
 * y_pred = K.clip(y_pred, K.epsilon(), 1)
 * and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)
 * @param ev
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class KullbackLeiblerDivergenceCriterion[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private val epsilon: T = ev.fromType(1e-07)

  override def updateOutput(input: Tensor[T], target : Tensor[T]): T = {
    val buffer1 = Tensor[T]()
    val buffer2 = Tensor[T]()
    buffer1.resizeAs(target).copy(target)
    buffer2.resizeAs(input).copy(input)

    buffer1.apply1(e => ev.clip(e, epsilon, ev.fromType(1.0)))
    buffer2.apply1(e => ev.clip(e, epsilon, ev.fromType(1.0)))

    val mul = buffer1.div(buffer2).log().cmul(buffer1).mean()
    mul
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {

    gradInput.resizeAs(input).copy(input)
    val gradArray = gradInput.storage().array()
    val gradOffset = gradInput.storageOffset() - 1
    val targetArray: Array[T] = target.storage().array()
    val targetOffset = target.storageOffset() - 1

    var i = 0
    while(i < gradInput.nElement()) {
      val z = gradArray(i + gradOffset)
      val y = targetArray(i + targetOffset)
      gradArray(i + gradOffset) = ev.plus(y, ev.negative(z))
      i += 1
    }
    gradInput
  }
}

object KullbackLeiblerDivergenceCriterion {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): KullbackLeiblerDivergenceCriterion[T]
  = new KullbackLeiblerDivergenceCriterion[T]()
}