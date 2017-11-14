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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc4, TensorFunc6}

import scala.reflect.ClassTag

/**
 * Creates a criterion that optimizes a two-class classification (squared)
 * hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.
 *
 * When margin = 1, sizeAverage = True and squared = False, this is the same as hinge loss in keras;
 * When margin = 1, sizeAverage = False and squared = True, this is the same as squared_hinge loss
 * in keras.
 *
 * @param margin if unspecified, is by default 1.
 * @param sizeAverage whether to average the loss
 * @param squared whether to calculate the squared hinge loss
 */

@SerialVersionUID( - 5028892499250398130L)
class MarginCriterion[@specialized(Float, Double) T: ClassTag]
 (val margin: Double = 1.0, val sizeAverage: Boolean = true, squared: Boolean = false)
 (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    var sum: T = ev.fromType(0)
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
        val z = ev.minus(ev.fromType(margin), ev.times(data1(index1), data2(index2)))
        if (ev.isGreater(z, ev.fromType(0))) {
          if (squared) {
            sum = ev.plus(sum, ev.times(z, z))
          } else {
            sum = ev.plus(sum, z)
          }
        }
      }
    }
    DenseTensorApply.apply2[T](input, target, func)
    if (sizeAverage) sum = ev.divide(sum, ev.fromType(input.nElement()))
    sum
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val norm = ev.fromType(if (sizeAverage) -1.0 / input.nElement() else -1.0)
    gradInput.resizeAs(input)

    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc6[T] {
      override def apply (data1: Array[T], offset1: Int, data2: Array[T],
                          offset2: Int, data3: Array[T], offset3: Int): Unit = {
        if (ev.isGreater(ev.fromType(margin), ev.times(data2(offset2), data3(offset3)))) {
          if (squared) {
            // dl/dx = -2y(1-xy)
            data1(offset1) = ev.times(
              ev.times(ev.times(ev.fromType(2), norm), data3(offset3)),
              ev.minus(ev.fromType(margin),
                ev.times(data2(offset2), data3(offset3))))
          } else {
            data1(offset1) = ev.times(norm, data3(offset3))
          }
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, input, target, func)
    gradInput
  }

  override def toString(): String = {
    s"nn.MarginCriterion($margin)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MarginCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MarginCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        margin == that.margin &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), margin, sizeAverage)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object MarginCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      margin: Double = 1.0,
      sizeAverage: Boolean = true,
      squared: Boolean = false)(implicit ev: TensorNumeric[T]) : MarginCriterion[T] = {
    new MarginCriterion[T](margin, sizeAverage, squared)
  }
}
