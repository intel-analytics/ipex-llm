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
 * It caculates diff = K.abs((y - x) / K.clip(K.abs(y), K.epsilon(), Double.MaxValue))
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
    buffer2.resizeAs(target).copy(target)
    buffer1.sub(target).abs()
    // buffer2 = K.clip(K.abs(y), K.epsilon(), Double.MaxValue)
    buffer2.apply1(e => ev.clip(ev.abs(e), epsilon, maxValue))
    buffer1.div(buffer2)

    output = ev.times(buffer1.mean(), ev.fromType(100.0))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val norm : T = ev.fromType(100.0 / input.nElement())

    buffer1.resizeAs(input).copy(input)
    gradInput.resizeAs(target).copy(target)

    val func = new TensorFunc6[T] {
      override def apply(inputBuf: Array[T], inputOffset: Int, targetClipBuf: Array[T],
        targetClipOffset: Int, gradInputBuf: Array[T], gradInputOffset: Int): Unit = {
        val a = inputBuf(inputOffset)
        val b = targetClipBuf(targetClipOffset)
        val c = gradInputBuf(gradInputOffset)

        if (a == c) {
          // x=y, gradInput value = 0
          gradInputBuf(gradInputOffset) = ev.zero
        } else if (ev.isGreater(a, c)) {
          // x > y, gradInput value = 1/K.clip(K.abs(y), K.epsilon(), Double.MaxValue)
          gradInputBuf(gradInputOffset) = ev.divide(ev.one, b)
        } else {
          // x < y, , gradInput value = -1/K.clip(K.abs(y), K.epsilon(), Double.MaxValue)
          gradInputBuf(gradInputOffset) = ev.divide(negativeOne, b)
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
