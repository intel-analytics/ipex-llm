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
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

/**
 * This method is same as `mean_squared_logarithmic_error` loss in keras.
 * It calculates:
 * first_log = K.log(K.clip(y, K.epsilon(), Double.MaxValue) + 1.)
 * second_log = K.log(K.clip(x, K.epsilon(), Double.MaxValue) + 1.)
 * and output K.mean(K.square(first_log - second_log))
 * Here, the x and y can have or not have a batch.
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class MeanSquaredLogarithmicCriterion[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private val epsilon: T = ev.fromType(1e-07)
  private val maxValue: T = ev.fromType(Double.MaxValue)

  @transient
  private var buffer1: Tensor[T] = null // first_log
  @transient
  private var buffer2: Tensor[T] = null // second_log

  override def updateOutput(input: Tensor[T], target : Tensor[T]): T = {
    if (buffer1 == null) buffer1 = Tensor[T]()
    if (buffer2 == null) buffer2 = Tensor[T]()
    buffer1.resizeAs(target).copy(target)
    buffer2.resizeAs(input).copy(input)

    buffer1.apply1(e => ev.clip(e, epsilon, ev.fromType(Double.MaxValue)))
    buffer1.add(ev.one).log()

    buffer2.apply1(e => ev.clip(e, epsilon, ev.fromType(Double.MaxValue)))
    buffer2.add(ev.one)
    // keep result K.clip(x K.epsilon(), Double.MaxValue) + 1.
    gradInput.resizeAs(buffer2).copy(buffer2)
    buffer2.log()

    buffer1.sub(buffer2)
    buffer2.copy(buffer1) // keep result of (first_log - second_log)
    output = buffer1.square().mean()
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val norm : Double = -2.0 / input.nElement()
    buffer2.mul(ev.fromType(norm))
    buffer2.div(gradInput)

    gradInput.resizeAs(input).copy(input)
    val gradArray = gradInput.storage().array()
    val gradOffset = gradInput.storageOffset() - 1
    val bufferArray = buffer2.storage().array()
    val bufferOffset = buffer2.storageOffset() - 1

    var i = 0
    while(i < gradInput.nElement()) {
      val z = gradArray(i + gradOffset)
      gradArray(i + gradOffset) = if (ev.isGreaterEq(z, epsilon) && ev.isGreaterEq(maxValue, z)) {
        bufferArray(i + bufferOffset)
      } else {
        ev.zero
      }
      i += 1
    }
    gradInput
  }
}

object MeanSquaredLogarithmicCriterion {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): MeanSquaredLogarithmicCriterion[T]
  = new MeanSquaredLogarithmicCriterion[T]()
}
