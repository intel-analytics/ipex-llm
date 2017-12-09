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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This method is same as `kullback_leibler_divergence` loss in keras.
 * Loss calculated as:
 * y_true = K.clip(y_true, K.epsilon(), 1)
 * y_pred = K.clip(y_pred, K.epsilon(), 1)
 * and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class KullbackLeiblerDivergenceCriterion[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private val epsilon: T = ev.fromType(1e-7)
  private val upperlimit = ev.fromType(1.0)

  @transient var bufferInput: Tensor[T] = null
  @transient var bufferTarget: Tensor[T] = null

  /**
   * It calculates:
   * y_true = K.clip(y_true, K.epsilon(), 1)
   * y_pred = K.clip(y_pred, K.epsilon(), 1)
   * and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)
   */
  override def updateOutput(input: Tensor[T], target : Tensor[T]): T = {
    if (bufferInput == null) bufferInput = Tensor[T]()
    if (bufferTarget == null) bufferTarget = Tensor[T]()
    require(input.isSameSizeAs(target),
      s"Input should have the same size as target. input size: (${input.size().mkString(", ")});" +
        s" target size: (${target.size().mkString(", ")}).")

    bufferInput.resizeAs(input).copy(input)
    bufferTarget.resizeAs(target).copy(target)
    bufferInput.apply1(e => ev.clip(e, epsilon, upperlimit))
    bufferTarget.apply1(e => ev.clip(e, epsilon, upperlimit))
    gradInput = bufferTarget.clone().div(bufferInput)

    // use bufferInput hold the intermediate value
    bufferInput.copy(gradInput)
    val mul = bufferInput.log().cmul(bufferTarget).sum()
    val batchSize = if (input.nDimension() == 1) 1 else input.size(1)
    ev.divide(mul, ev.fromType(batchSize))
  }

  /**
   * back propagation with: - target / input
   */
  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(target),
      s"Input should have the same size as target. input size: (${input.size().mkString(", ")});" +
        s" target size: (${target.size().mkString(", ")}).")

    val batchSize = if (input.nDimension() == 1) 1 else input.size(1)
    gradInput.div(ev.fromType(-batchSize))

    // keep consistent with Keras for values out of clip boundary
    val func1 = new TensorFunc6[T] {
      private val nonGradient = ev.fromType(0)
      override def apply(
          gradInputBuf: Array[T], gradInputOffset: Int,
          inputBuf: Array[T], InputOffset: Int,
          targetBuf: Array[T], targetOffset: Int
          ): Unit = {
        if (ev.isGreater(inputBuf(InputOffset), upperlimit)
          && ev.isGreater(targetBuf(targetOffset), upperlimit)) {
          gradInputBuf(gradInputOffset) = nonGradient
        } else if (ev.isGreater(epsilon, inputBuf(InputOffset))) {
          gradInputBuf(gradInputOffset) = nonGradient
        }
      }
    }

    DenseTensorApply.apply3[T](gradInput, input, target, func1)
    gradInput
  }
}

object KullbackLeiblerDivergenceCriterion {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): KullbackLeiblerDivergenceCriterion[T] =
    new KullbackLeiblerDivergenceCriterion[T]()
}
