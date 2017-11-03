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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * a smooth version of the AbsCriterion
 * It uses a squared term if the absolute element-wise error falls below 1.
 * It is less sensitive to outliers than the MSECriterion and in some cases
 * prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
 *
 * d = (x - y) * w_in
 * loss(x, y, w_in, w_out)
 *            | 0.5 * (sigma * d_i)^2 * w_out          if |d_i| < 1 / sigma / sigma
 * = 1/n \sum |
 *            | (|d_i| - 0.5 / sigma / sigma) * w_out   otherwise
 * @tparam T
 */
class SmoothL1CriterionWithWeights[@specialized(Float, Double) T: ClassTag]
(val sigma: Double, val num: Int = 0)
  (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Tensor[T], Table, T] {
  @transient var buffer: Tensor[T] = _
  // diff holds (input - gt) * w_in
  @transient var diff: Tensor[T] = _
  val sigma2 = sigma * sigma
  @transient var hasWeights = true

  override def updateOutput(input: Tensor[T], target: Table): T = {
    // the target are composed of gt, inside_weight, outside_weight
    assert(target.length() >= 1)
    val bboxTarget = target[Tensor[T]](1)
    var insideW: Tensor[T] = null
    var outsideW: Tensor[T] = null
    if (target.length() == 1) {
      hasWeights = false
      require(input.nElement() == bboxTarget.nElement(), s" " +
        s"the length of bbox target, " +
        s"input must be equal, input length ${input.nElement()}," +
        s" bbox target length ${bboxTarget.nElement()}")
    } else {
      hasWeights = true
      insideW = target[Tensor[T]](2)
      outsideW = target[Tensor[T]](3)
      require(insideW.nElement() == outsideW.nElement() &&
        insideW.nElement() == bboxTarget.nElement(),
        s"the length of bbox target, insideW, outsideW must be equal, " +
          s"bbox target ${bboxTarget.nElement()}," +
          s"insideW ${insideW.nElement()}," +
          s"outsideW ${outsideW.nElement()}")
    }

    if (diff == null) {
      diff = Tensor[T]()
    }
    // input - gt
    diff.add(input, ev.fromType(-1), bboxTarget)
    if (hasWeights) {
      // apply "inside" weights, (input - gt) * w_in
      diff.cmul(insideW)
    }


    if (buffer == null) {
      buffer = Tensor[T]
    }
    // |input - gt| * w_in
    buffer.resizeAs(diff).copy(diff).abs()
    val data = buffer.storage().array()
    val dataOffset = buffer.storageOffset() - 1
    var i = 0
    while (i < buffer.nElement()) {
      // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
      //        |x| - 0.5 / sigma / sigma    otherwise
      if (ev.isGreater(ev.fromType(1.0 / sigma2), data(dataOffset + i))) {
        data(dataOffset + i) = ev.times(ev.fromType[Double](sigma2),
          ev.times(ev.fromType(0.5), ev.times(data(dataOffset + i), data(dataOffset + i))))
      }
      else {
        data(dataOffset + i) = ev.minus(data(dataOffset + i), ev.fromType[Double](0.5 / sigma2))
      }
      i += 1
    }
    if (hasWeights) {
      // apply "outside" weights,  w_out * SmoothL1(|input - gt| * w_in)
      buffer.cmul(outsideW)
    }
    output = if (num > 0) {
      ev.divide(buffer.sum(), ev.fromType(num))
    } else {
      ev.divide(buffer.sum(), ev.fromType(input.size(1)))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Table): Tensor[T] = {
    assert(target.length() >= 1)

    val bboxTarget = target[Tensor[T]](1)
    var insideW: Tensor[T] = null
    var outsideW: Tensor[T] = null
    if (target.length() == 1) {
      hasWeights = false
      require(input.nElement() == bboxTarget.nElement(),
        "the length of bbox target, input must be equal, " +
          s"input length ${input.nElement()}, " +
          s"bbox target length ${bboxTarget.nElement()}")
    } else {
      hasWeights = true
      insideW = target[Tensor[T]](2)
      outsideW = target[Tensor[T]](3)
      require(insideW.nElement() == outsideW.nElement() &&
        insideW.nElement() == bboxTarget.nElement(),
        "the length of bbox target, insideW, outsideW must be equal, " +
          s"bbox target ${bboxTarget.nElement()}," +
          s"insideW ${insideW.nElement()}," +
          s"outsideW ${outsideW.nElement()}")
    }
    val data = diff.storage().array()
    val dataOffset = diff.storageOffset() - 1
    var i = 0
    while (i < diff.nElement()) {
      // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
      //       = sign(x)
      val x = data(dataOffset + i)
      if (ev.isGreater(ev.fromType[Double](1.0 / sigma2), ev.abs(x))) {
        data(dataOffset + i) = ev.times(ev.fromType[Double](sigma2), x)
      } else {
        // sign(x) == (0<x) - (x<0)
        if (ev.isGreater(data(dataOffset + i), ev.fromType(0))) {
          data(dataOffset + i) = ev.fromType(1)
        } else if (ev.isGreater(ev.fromType(0), data(dataOffset + i))) {
          data(dataOffset + i) = ev.fromType(-1)
        } else {
          data(dataOffset + i) = ev.fromType(0)
        }
      }
      i += 1
    }

    gradInput.resizeAs(diff).copy(diff)
    if (num > 0) {
      gradInput.div(ev.fromType(num))
    } else {
      gradInput.div(ev.fromType(input.size(1)))
    }
    if (hasWeights) {
      // scale by inside weight
      gradInput.cmul(insideW)
      // scale by outside weight
      gradInput.cmul(outsideW)
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.SmoothL1CriterionWithWeights"
  }
}

object SmoothL1CriterionWithWeights {
  def apply[@specialized(Float, Double) T: ClassTag](sigma: Double, num: Int = 0)
    (implicit ev: TensorNumeric[T]): SmoothL1CriterionWithWeights[T] =
    new SmoothL1CriterionWithWeights[T](sigma, num)
}
