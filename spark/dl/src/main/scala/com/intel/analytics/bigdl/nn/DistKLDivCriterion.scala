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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc4}

import scala.reflect.ClassTag

/**
 * The Kullback–Leibler divergence criterion
 * @param sizeAverage
 */

@SerialVersionUID(5018120506588694055L)
class DistKLDivCriterion[@specialized(Float, Double) T: ClassTag](val sizeAverage: Boolean = true)
 (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require((input.dim() == target.dim()) && (input.isSameSizeAs(target)),
      "DistKLDivCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget)

    var sum: T = ev.zero
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
        if (ev.isGreater(data2(index2), ev.zero)) {
          sum = ev.plus(sum, ev.times(data2(index2),
            ev.minus(ev.log(data2(index2)), data1(index1))))
        }
      }
    }
    DenseTensorApply.apply2[T](input, target, func)
    if (sizeAverage) sum = ev.divide(sum, ev.fromType(input.nElement()))
    sum
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require((input.dim() == target.dim()) && (input.isSameSizeAs(target)),
      "DistKLDivCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget)

    val norm = ev.fromType(if (sizeAverage) -1.0 / input.nElement() else -1.0)
    gradInput.resizeAs(input)
    // todo: the performance of contiguous tensor should be optimized
    val func = new TensorFunc4[T] {
      override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
        if (ev.isGreater(data2(index2), ev.zero)) {
          data1(index1) = ev.times(data2(index2), norm)
        }
      }
    }
    DenseTensorApply.apply2[T](gradInput, target, func)
    gradInput
  }

  override def toString(): String = {
    s"nn.DistKLDivCriterion ($sizeAverage)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[DistKLDivCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: DistKLDivCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), sizeAverage)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object DistKLDivCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : DistKLDivCriterion[T] = {
    new DistKLDivCriterion[T](sizeAverage)
  }
}
