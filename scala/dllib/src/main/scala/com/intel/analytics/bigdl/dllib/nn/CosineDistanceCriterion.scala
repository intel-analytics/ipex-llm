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
 * Creates a criterion that measures the loss given an input tensor and target tensor.
 *
 * The input and target are two tensors with same size.
 * For instance:
 *
 * x = Tensor[Double](Storage(Array(0.1, 0.2, 0.3)))
 * y = Tensor[Double](Storage(Array(0.15, 0.25, 0.35)))
 *
 * loss(x, y) = 1 - cos(x, y)
 */

@SerialVersionUID(- 4008475267198411701L)
class CosineDistanceCriterion[@specialized(Float, Double) T: ClassTag]
(val sizeAverage: Boolean = true)
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  @transient
  private var buffer: Tensor[T] = null
  @transient
  private var w1: Tensor[T] = null
  @transient
  private var w22: Tensor[T] = null
  @transient
  private var w: Tensor[T] = null
  @transient
  private var w32: Tensor[T] = null
  @transient
  private var _outputs: Tensor[T] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    var input1 = input
    var input2 = target

    if (null == buffer) buffer = Tensor[T]()
    if (null == w1) w1 = Tensor[T]()
    if (null == w22) w22 = Tensor[T]()
    if (null == w) w = Tensor[T]()
    if (null == _outputs) _outputs = Tensor[T]()
    if (null == w32) w32 = Tensor[T]()

    if (input1.dim() == 1) {
      input1 = input1.view(1, input1.nElement())
      input2 = input2.view(1, input2.nElement())
    }

    buffer.resizeAs(input1).cmul(input1, input2)
    w1.sum(buffer, 2)

    val epsilon = 1e-12
    buffer.cmul(input1, input1)
    w22.sum(buffer, 2).add(ev.fromType(epsilon))
    _outputs.resizeAs(w22).fill(ev.fromType(1))
    w22.cdiv(_outputs, w22)
    w.resizeAs(w22).copy(w22)

    buffer.cmul(input2, input2)
    w32.sum(buffer, 2).add(ev.fromType(epsilon))
    w32.cdiv(_outputs, w32)
    w.cmul(w32)
    w.sqrt()

    _outputs.cmul(w1, w)
    _outputs.mul(ev.fromType(-1)).add(ev.fromType(1))
    output = _outputs.sum()

    if (sizeAverage) {
      output = ev.divide(output, ev.fromType(input.size(1)))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    var v1 = input
    var v2 = target

    if (v1.dim() == 1) {
      v1 = v1.view(1, v1.nElement())
      v2 = v2.view(1, v2.nElement())
    }

    if (null == gradInput) gradInput = Tensor[T]()

    val gw1 = gradInput

    gw1.resizeAs(v1).copy(v2)

    buffer.resizeAs(w1).cmul(w1, w22)
    gw1.addcmul(ev.fromType(-1), buffer.expandAs(v1), v1)
    gw1.cmul(w.expandAs(v1)).mul(ev.fromType(-1))

    if (sizeAverage) {
      gradInput.div(ev.fromType(v2.size(1)))
    }

    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CosineEmbeddingCriterion[T]]

  override def toString(): String = {
    s"nn.CosineEmbeddingCriterion($sizeAverage)"
  }

  override def equals(other: Any): Boolean = other match {
    case that: CosineDistanceCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), sizeAverage)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object CosineDistanceCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : CosineDistanceCriterion[T] = {
    new CosineDistanceCriterion[T](sizeAverage)
  }
}
