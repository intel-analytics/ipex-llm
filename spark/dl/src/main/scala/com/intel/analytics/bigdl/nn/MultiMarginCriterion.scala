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
 * Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss)
 * between input x and output y (which is a target class index).
 *
 * @param p
 * @param weights
 * @param margin
 * @param sizeAverage
 */

@SerialVersionUID(- 5690966547738588572L)
class MultiMarginCriterion[@specialized(Float, Double) T: ClassTag](val p: Int = 1,
 val weights: Tensor[T] = null, margin: Double = 1.0, val sizeAverage: Boolean = true)
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  require(p == 1 || p == 2, s"MultiMarginCriterion: only p=1 and p=2 supported, but get p $p")
  if (null != weights) {
    require(weights.dim() == 1, s"MultiMarginCriterion: weights input should be 1-D Tensor, " +
      s"but get weights dim ${weights.dim()}")
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.nDimension() == 1 || input.nDimension() == 2,
    "MultiMarginCriterion: " +
      ErrorInfo.constrainInputAsVectorOrBatch +
      s"input dimension ${input.nDimension()}")

    val (nframe, dim) = if (input.nDimension() == 1) {
      (1, input.size(1))
    } else {
      require(target.nDimension() == 1 && target.size(1) == input.size(1),
      "MultiMarginCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget)
      (input.size(1), input.size(2))
    }

    require(ev.isGreaterEq(target.min(), ev.fromType(0)) &&
      ev.isGreaterEq(ev.fromType(dim), target.max()), "MultiMarginCriterion: " +
      s"target out of range, target min should be greater than or equal to zero, but get " +
      s"${target.min()}, target max should be less than or equal to $dim, but get ${target.max()}")

    val _target = target.contiguous()
    val _input = input.contiguous()
    val _weights = if (null != weights) weights.contiguous() else null

    val input_data = _input.storage().array()
    val target_data = _target.storage().array()
    val weights_data = if (null != _weights) _weights.storage().array() else null

    val input_offset = _input.storageOffset() - 1
    val target_offset = _target.storageOffset() - 1
    val weights_offset = if (null != _weights) _weights.storageOffset() - 1 else 0

    var sum: T = ev.fromType(0)
    var t = 0
    var n = 0
    while (t < nframe) {
      val target_idx = ev.toType[Int](target_data(t + target_offset)) - 1
      val input_target = input_data(n + target_idx + input_offset)
      var d = 0
      while (d < dim) {
        val z = ev.plus(ev.minus(ev.fromType(margin), input_target),
          input_data(n + d + input_offset))
        if ((d != target_idx) && (ev.isGreater(z, ev.fromType(0)))) {
          var h = if (p == 1) z else ev.times(z, z)
          if (null != weights_data) h = ev.times(h, weights_data(target_idx + weights_offset))
          sum = ev.plus(sum, h)
        }
        d += 1
      }
      t += 1
      n += dim
    }

    sum = ev.divide(sum, ev.fromType(dim))
    if (sizeAverage) sum = ev.divide(sum, ev.fromType(nframe))
    sum
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 1 || input.nDimension() == 2,
    "MultiMarginCriterion: " +
      ErrorInfo.constrainInputAsVectorOrBatch +
      s"input dimension ${input.nDimension()}")
    val (nframe, dim) = if (input.nDimension() == 1) {
      (1, input.size(1))
    } else {
      require(target.nDimension() == 1 && target.size(1) == input.size(1),
      "MultiMarginCriterion: " +
        ErrorInfo.constrainInputSizeSameAsTarget +
        s"target dimension ${target.nDimension()}, " +
        s"target size[1] ${target.size(1)}, " +
        s"input size[1] ${input.size(1)}")
      (input.size(1), input.size(2))
    }
    val g = ev.fromType(if (sizeAverage)  1.0/(nframe*dim) else 1.0/(dim))

    val _target = target.contiguous()
    val _input = input.contiguous()
    val _weights = if (null != weights) weights.contiguous() else null

    val input_data = _input.storage().array()
    val target_data = _target.storage().array()
    val weights_data = if (null != _weights) _weights.storage().array() else null

    val input_offset = _input.storageOffset() - 1
    val target_offset = _target.storageOffset() - 1
    val weights_offset = if (null != _weights) _weights.storageOffset() - 1 else 0

    gradInput.resizeAs(input).zero()
    val gradInput_data = gradInput.storage().array()

    var t = 0
    var n = 0
    while (t < nframe) {
      val target_idx = ev.toType[Int](target_data(t + target_offset)) - 1
      val input_target = input_data(n + target_idx + input_offset)
      var gradInput_target = ev.fromType(0)

      var d = 0
      while (d < dim) {
        val z = ev.plus(ev.minus(ev.fromType(margin), input_target), input_data(n + d))
        if (d != target_idx) {
          if (ev.isGreater(z, ev.fromType(0))) {
            var h = if (p == 1) g else ev.times(ev.fromType(2), ev.times(g, z))
            if (null != weights_data) h = ev.times(h, weights_data(target_idx + weights_offset))
            gradInput_target = ev.minus(gradInput_target, h)
            gradInput_data(n + d) = h
          } else {
            gradInput_data(n + d) = ev.fromType(0)
          }
        }
        d += 1
      }
      gradInput_data(n + target_idx) = gradInput_target
      n += dim
      t += 1
    }
    gradInput
  }


  override def toString(): String = {
    s"nn.MultiMarginCriterion($sizeAverage, $weights, $margin)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MultiMarginCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MultiMarginCriterion[T] =>
      super.equals(that) &&
      (that canEqual this) &&
        p == that.p &&
        weights == that.weights &&
        sizeAverage == that.sizeAverage
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), p, weights, sizeAverage)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object MultiMarginCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      p: Int = 1,
      weights: Tensor[T] = null,
      margin: Double = 1.0,
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : MultiMarginCriterion[T] = {
    new MultiMarginCriterion[T](p, weights, margin, sizeAverage)
  }
}
