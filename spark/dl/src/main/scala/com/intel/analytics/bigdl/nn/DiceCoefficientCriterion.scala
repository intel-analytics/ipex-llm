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
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

/**
 * The Dice-Coefficient criterion
 * input: Tensor, target: Tensor
 *
 * return:      2 * (input intersection target)
 *         1 - ----------------------------------
 *                input union target
 *
 * @param sizeAverage
 * @param epsilon  small offset
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */

@SerialVersionUID(- 1446868477754414191L)
class DiceCoefficientCriterion[@specialized(Float, Double) T: ClassTag]
 (val sizeAverage: Boolean = true, val epsilon: Float = 1.0f)
 (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  @transient
  private val buffer1: Tensor[T] = Tensor[T]()
  @transient
  private val buffer2: Tensor[T] = Tensor[T]()
  @transient
  private val buffer3: Tensor[T] = Tensor[T]()
  @transient
  private val w1: Tensor[T] = Tensor[T]()
  @transient
  private val w2: Tensor[T] = Tensor[T]()
  @transient
  private val _outputs: Tensor[T] = Tensor[T]()

  @transient
  private var _input: Tensor[T] = null
  @transient
  private var _target: Tensor[T] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require((input.dim() == target.dim()) && (input.isSameSizeAs(target)),
      "DiceCoefficientCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget +
    s"input (${input.dim()}) target(${target.dim()})")

    _input = if (input.dim() == 1) input.view(1, input.nElement()) else input
    _target = if (target.dim() == 1) target.view(1, target.nElement()) else target

    /**
     * w1 = 2 \sum(x_i * y_i) + \epsilon
     */
    buffer1.resizeAs(_input)
    buffer1.cmul(_input, _target)
    w1.sum(buffer1, 2)
    _outputs.resizeAs(w1).fill(ev.fromType(2))
    w1.cmul(_outputs).add(ev.fromType(epsilon))

    /**
     * w2 = \sum(x_i) + \sum(y_i) + \epsilon
     */
    buffer1.sum(_input, 2)
    w2.sum(_target, 2).add(buffer1).add(ev.fromType(epsilon))

    /**
     * Loss = 1 - w1 / w2
     */
    _outputs.cdiv(w1, w2).mul(ev.fromType(-1)).add(ev.fromType(1))
    output = _outputs.sum()

    if (sizeAverage) {
      output = ev.divide(output, ev.fromType(_target.size(1)))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {

    /**
     * buffer1 = w1 / w2*w2
     * buffer2 = 1 / w2
     *
     * gradInput = - 2 * target / w2 + w1 / w2 * w2
     */
    buffer2.resizeAs(w2)
    buffer2.cmul(w2, w2)
    buffer3.resizeAs(w1)
    buffer3.cdiv(w1, buffer2)

    _outputs.resizeAs(w2).fill(ev.fromType(1))
    buffer2.cdiv(_outputs, w2)

    gradInput.resizeAs(_input)
    gradInput.addcmul(ev.fromType(-2), buffer2.expandAs(_input), _target)
    gradInput.add(buffer3.expandAs(_input))

    if (sizeAverage) {
      gradInput.div(ev.fromType(_target.size(1)))
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.DiceCoefficientCriterion"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[DistKLDivCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: DiceCoefficientCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        epsilon == that.epsilon &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), epsilon, sizeAverage)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object DiceCoefficientCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    sizeAverage: Boolean = true, epsilon: Float = 1.0f)
    (implicit ev: TensorNumeric[T]) : DiceCoefficientCriterion[T] = {
    new DiceCoefficientCriterion[T](sizeAverage, epsilon)
  }
}
