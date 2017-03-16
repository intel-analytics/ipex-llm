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
 * This class is intended to support inputs with 3 or more dimensions.
 * Apply Any Provided Criterion to every temporal slice of an input.
 * @param critrn
 */

class TimeDistributedCriterion[T : ClassTag](
  val critrn : TensorCriterion[T],
  val sizeAverage: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private val fInput: Tensor[T] = Tensor[T]()
  private val fTarget: Tensor[T] = Tensor[T]()
  private var inputSize: Array[Int] = _
  private var targetSize: Array[Int] = _

  private def combine(src: Array[Int], target: Array[Int]): Unit = {
    require(src.length == target.length + 1,
      "TimeDistributedCriterion: combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")
    target(0) = src(0) * src(1)
    var j = 1
    while (j < target.length) {
      target(j) = src(j + 1)
      j += 1
    }
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    /**
     * For example
     * input.size = [B, T, D] => inputSize = [B * T, D]
     * target.size = [B, T] => targetSize = [B * T]
     */
    if (inputSize == null) {
      inputSize = new Array[Int](input.size.length - 1)
    }
    if (targetSize == null) {
      targetSize = new Array[Int](target.size.length - 1)
    }

    combine(input.size, inputSize)
    combine(target.size, targetSize)
    fInput.set(input).resize(inputSize)
    fTarget.set(target).resize(targetSize)
    output = critrn.updateOutput(fInput, fTarget)
    if (!sizeAverage) {
      output = ev.times(output, ev.fromType[Int](input.size(2)))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val _gradInput = critrn.updateGradInput(fInput, fTarget).toTensor[T]
    gradInput = _gradInput.resize(input.size)
    if (!sizeAverage) {
      gradInput.apply1(x => ev.times(x, ev.fromType[Int](input.size(2))))
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[TimeDistributedCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TimeDistributedCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        fInput == that.fInput &&
        fTarget == that.fTarget &&
        inputSize == that.inputSize &&
        targetSize == that.targetSize
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), fInput, fTarget, inputSize, targetSize)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object TimeDistributedCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    critrn: TensorCriterion[T] = null, sizeAverage: Boolean = false)
    (implicit ev: TensorNumeric[T]) : TimeDistributedCriterion[T] = {
    new TimeDistributedCriterion[T](critrn, sizeAverage)
  }
}
