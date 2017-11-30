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

  val bufferInput = Tensor[T]()
  val bufferTarget = Tensor[T]()

  /**
   * It calculates:
   * y_true = K.clip(y_true, K.epsilon(), 1)
   * y_pred = K.clip(y_pred, K.epsilon(), 1)
   * and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)
   */
  override def updateOutput(input: Tensor[T], target : Tensor[T]): T = {
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
          data1: Array[T], offset1: Int,
          data2: Array[T], offset2: Int,
          data3: Array[T], offset3: Int
          ): Unit = {
        if (ev.isGreater(data2(offset2), upperlimit) && ev.isGreater(data3(offset3), upperlimit)) {
          data1(offset1) = nonGradient
        } else if (ev.isGreater(epsilon, data2(offset2))) {
          data1(offset1) = nonGradient
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
