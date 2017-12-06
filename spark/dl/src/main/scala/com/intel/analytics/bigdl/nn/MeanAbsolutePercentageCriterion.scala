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
 * This method is same as `mean_absolute_percentage_error` loss in keras.
 * It caculates diff = K.abs((x - y) / K.clip(K.abs(x), K.epsilon(), Double.MaxValue))
 * and return 100 * K.mean(diff) as outpout
 * Here, the x and y can have or not have a batch.
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class MeanAbsolutePercentageCriterion[T: ClassTag]
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  @transient
  private var buffer1 : Tensor[T] = null
  @transient
  private var buffer2 : Tensor[T] = null
  private val epsilon: T = ev.fromType(1e-07)
  private val maxValue: T = ev.fromType(Double.MaxValue)
  private val negativeOne: T = ev.fromType(-1)


  override def updateOutput(input: Tensor[T], target : Tensor[T]): T = {
    if (buffer1 == null) buffer1 = Tensor[T]()
    if (buffer2 == null) buffer2 = Tensor[T]()
    buffer1.resizeAs(input).copy(input)
    buffer2.resizeAs(input).copy(input)
    buffer1.sub(target).abs()
    buffer2.apply1(e => ev.clip(ev.abs(e), epsilon, maxValue))
    buffer1.div(buffer2)

    output = ev.times(buffer1.mean(), ev.fromType(100.0))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val norm : T = ev.fromType(100.0 / input.nElement())

    buffer1.resizeAs(input).copy(input)
    buffer2.resizeAs(target).copy(target)
    gradInput.resizeAs(input)

    val func = new TensorFunc6[T] {
      override def apply(inputBuf: Array[T], inputOffset: Int, targetBuf: Array[T],
                         targetOffset: Int, gradInputBuf: Array[T], gradInputOffset: Int): Unit = {
        val a = inputBuf(inputOffset)
        val b = targetBuf(targetOffset)

        if (a == b) {
          gradInputBuf(gradInputOffset) = ev.zero
        } else {
          // default calculate results of (x>y). results of (x<y) are negative values of (x>y)
          if (ev.isGreaterEq(ev.abs(a), epsilon) && ev.isGreaterEq(maxValue, ev.abs(a))) {
            val v = ev.divide(b, ev.times(a, a)) // y/x^2
            if (ev.isGreater(a, ev.zero)) {
              gradInputBuf(gradInputOffset) = v // y/x^2
            } else {
              gradInputBuf(gradInputOffset) = ev.times(v, negativeOne) // -y/x^2
            }
          } else if (ev.isGreater(epsilon, ev.abs(a))) {
            gradInputBuf(gradInputOffset) = ev.divide(ev.one, epsilon) // 1/epsilon
          } else {
            gradInputBuf(gradInputOffset) = ev.divide(ev.one, maxValue) // 1/Double.MaxValue
          }

          if (ev.isGreater(b, a)) { // x < y
            gradInputBuf(gradInputOffset) = ev.times(gradInputBuf(gradInputOffset), negativeOne)
          }
        }
      }
    }
    DenseTensorApply.apply3(buffer1, buffer2, gradInput, func)

    gradInput.mul(norm)
    gradInput
  }


  override def canEqual(other: Any): Boolean =
    other.isInstanceOf[MeanAbsolutePercentageCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MeanAbsolutePercentageCriterion[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode())
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object MeanAbsolutePercentageCriterion {
  def apply[T : ClassTag]()(implicit ev: TensorNumeric[T]): MeanAbsolutePercentageCriterion[T]
  = new MeanAbsolutePercentageCriterion[T]()
}
