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
 * This criterion combines LogSoftMax and ClassNLLCriterion in one single class.
 *
 * @param weights A tensor assigning weight to each of the classes
 */

@SerialVersionUID(- 5446858218997354022L)
class CrossEntropyCriterion[T: ClassTag](
  val weights: Tensor[T] = null,
  val sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  private val lsm = new LogSoftMax[T]()
  private val nll = new ClassNLLCriterion[T](weights, sizeAverage)

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    lsm.updateOutput(input)
    nll.updateOutput(lsm.output, target.asInstanceOf[Tensor[T]])
    output = nll.output
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val size = input.size()
    var _gradInput = Tensor[T]()
    _gradInput = nll.updateGradInput(lsm.output, target)
    lsm.updateGradInput(input, _gradInput)
    gradInput.resizeAs(lsm.gradInput).copy(lsm.gradInput).view(size)
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CrossEntropyCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: CrossEntropyCriterion[T] =>
      (that canEqual this) &&
        weights == that.weights &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), weights)
    state.map(getHashCode).foldLeft(0)((a, b) => 37 * a + b)
  }

  override def toString(): String = {
    s"nn.CrossEntropyCriterion"
  }
}

object CrossEntropyCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    weights: Tensor[T] = null, sizeAverage: Boolean = true)
    (implicit ev: TensorNumeric[T]) : CrossEntropyCriterion[T] = {
    new CrossEntropyCriterion[T](weights, sizeAverage)
  }
}
