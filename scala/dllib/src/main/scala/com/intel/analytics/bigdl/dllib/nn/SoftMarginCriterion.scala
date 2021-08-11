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
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Creates a criterion that optimizes a two-class classification logistic loss
 * between input x (a Tensor of dimension 1) and output y (which is a tensor
 * containing either 1s or -1s).
 *
 *    loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x:nElement()
 *
 * @param sizeAverage The normalization by the number of elements in the input
 *                     can be disabled by setting
 */
@SerialVersionUID(7573077918688542348L)
class SoftMarginCriterion[@specialized(Float, Double) T: ClassTag](var sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T])
  extends TensorCriterion[T] {
  def isSizeAverage: Boolean = sizeAverage

  def setSizeAverage(sizeAverage: Boolean): this.type = {
    this.sizeAverage = sizeAverage
    this
  }

  // TODO: replace apply for performance optimization
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.isSameSizeAs(target), "The input should have the same size as target" +
      s"input size ${input.nElement()}, target size ${target.nElement()}")
    var sum = ev.zero
    val func2 = new TensorFunc4[T] {
      override def apply(in: Array[T], index1: Int, tar: Array[T], index2: Int): Unit = {
        val z = ev.log(ev.plus(ev.one, ev.exp(ev.negative(ev.times(in(index1), tar(index2))))))
        sum = ev.plus(sum, z)
      }
    }
    DenseTensorApply.apply2[T](input, target, func2)

    if (sizeAverage) {
      sum = ev.divide(sum, ev.fromType[Int](input.nElement()))
    }

    output = sum
    output
  }

  // TODO: replace apply for performance optimization
  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(target), "The input should have the same size as target" +
      s"input size ${input.nElement()}, target size ${target.nElement()}")
    val norm = if (sizeAverage) {
      ev.divide(ev.one, ev.fromType[Int](input.nElement()))
    } else {
      ev.one
    }

    gradInput.resizeAs(input)
    val func = new TensorFunc6[T] {
      override def apply (gradInput: Array[T], offset1: Int, input: Array[T],
        offset2: Int, target: Array[T], offset3: Int): Unit = {
        val z = ev.exp(ev.negative(ev.times(target(offset1), input(offset2))))
        gradInput(offset1) = ev.divide(
          ev.negative(ev.times(norm, ev.times(target(offset3), z))), ev.plus(ev.one, z))
      }
    }
    DenseTensorApply.apply3[T](gradInput, input, target, func)
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[SoftMarginCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: SoftMarginCriterion[T] =>
      (that canEqual this) &&
        gradInput == that.gradInput &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashcode(state: Any): Int = if (state == null) 0 else state.hashCode()
    val state = Seq(gradInput, sizeAverage)
    state.map(getHashcode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object SoftMarginCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true)
      (implicit ev: TensorNumeric[T]): SoftMarginCriterion[T] = {
    new SoftMarginCriterion(sizeAverage)
  }
}
