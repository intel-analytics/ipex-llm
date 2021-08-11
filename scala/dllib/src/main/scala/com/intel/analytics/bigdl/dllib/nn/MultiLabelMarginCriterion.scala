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
 * Creates a criterion that optimizes a multi-class multi-classification hinge loss
 * (margin-based loss) between input x and output y (which is a Tensor of target class indices)
 *
 * @param sizeAverage
 */

@SerialVersionUID(9034717449427139574L)
class MultiLabelMarginCriterion[@specialized(Float, Double) T: ClassTag]
(val sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  @transient
  private var isTarget: Tensor[T] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    if (null == isTarget) isTarget = Tensor[T]()
    require(input.nDimension() == 1 || input.nDimension() == 2,
      "MultiLabelMarginCriterion: " + ErrorInfo.constrainInputAsVectorOrBatch +
    s"input dimension ${input.nDimension()}")
    val (nframe, dim) = if (input.nDimension() == 1) {
      require(target.nDimension() == 1 && target.size(1) == input.size(1),
        "MultiLabelMarginCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget +
      s"target dimension ${target.nDimension()}, " +
          s"target size ${target.size(1)}, input size ${input.size()}")
      (1, input.size(1))
    } else {
      require(target.nDimension() == 2 && target.size(1) == input.size(1) && target.size(2)
        == input.size(2), "MultiLabelMarginCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget)
      (input.size(1), input.size(2))
    }
    require(ev.isGreaterEq(target.min(), ev.zero) &&
      ev.isGreaterEq(ev.fromType(dim), target.max()), "MultiLabelMarginCriterion: " +
      s"target out of range, target min should be greater than or equal to zero, but get " +
      s"${target.min()}, target max should be less than or equal to $dim, but get ${target.max()}")

    val _target = target.contiguous()
    val _input = input.contiguous()

    val input_data = _input.storage().array()
    val target_data = _target.storage().array()
    val input_offset = _input.storageOffset() - 1
    val target_offset = _target.storageOffset() - 1

    isTarget.resizeAs(target).zero()
    val isTarget_data = isTarget.storage().array()

    var sum: T = ev.fromType(0)
    var t = 0
    var n = 0
    while (t < nframe) {
      var ddt = 0
      var dt = 0
      while (ddt < dim) {
        val target_idx = ev.toType[Int](target_data(n + ddt + target_offset)) - 1
        if(target_idx >= 0) {
          isTarget_data(n + target_idx) = ev.fromType(1)
          ddt += 1
        } else {
          ddt = dim
        }
      }

      while (dt < dim) {
        val  target_idx = ev.toType[Int](target_data(n + dt + target_offset)) - 1
        if (target_idx >= 0) {
          val input_target = input_data(n + target_idx + input_offset)
          var d = 0
          while (d < dim) {
            if (isTarget_data(n + d) == 0) {
              val z = ev.plus(ev.minus(ev.fromType(1), input_target),
                input_data(n + d + input_offset))
              if (ev.isGreater(z, ev.fromType(0))) sum = ev.plus(sum, z)
            }
            d += 1
          }
          dt += 1
        } else {
          dt = dim
        }
      }
      n += dim
      t += 1
    }

    sum = ev.divide(sum, ev.fromType(dim))
    if (sizeAverage) sum = ev.divide(sum, ev.fromType(nframe))
    sum
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 1 || input.nDimension() == 2,
      "MultiLabelMarginCriterion: " + ErrorInfo.constrainInputAsVectorOrBatch +
    s"input dimension ${input.nDimension()}")
    val (nframe, dim) = if (input.nDimension() == 1) {
      require(target.nDimension() == 1 && target.size(1) == input.size(1),
        "MultiLabelMarginCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget +
      s"target dimension ${target.nDimension()}" +
          s"target size ${target.size(1)} input size ${input.size(1)}")
      require(isTarget.nDimension() == 1 && isTarget.size(1) == input.size(1),
        "MultiLabelMarginCriterion: inconsistent isTarget size" +
          s"isTarget dimension ${isTarget.size(1)}" +
          s"isTarget size ${isTarget.size(1)} input size ${input.size(1)}")
      (1, input.size(1))
    } else {
      require(target.nDimension() == 2 && target.size(1) == input.size(1) && target.size(2)
        == input.size(2), "MultiLabelMarginCriterion: " + ErrorInfo.constrainInputSizeSameAsTarget +
      s"target dimension ${target.nDimension()} " +
        s"target size(${target.size(1)},${target.size(2)})" +
        s"input size(${input.size(1)},${input.size(2)})")
      require(isTarget.nDimension() == 2 && isTarget.size(1) == input.size(1) &&
        isTarget.size(2) == input.size(2), "MultiLabelMarginCriterion: inconsistent isTarget size" +
        s"isTarget dimension ${isTarget.nDimension()}" +
        s"isTarget size(${isTarget.size(1)},${isTarget.size(2)})" +
        s"input size(${input.size(1)},${input.size(2)})")
      (input.size(1), input.size(2))
    }

    require(ev.isGreaterEq(target.min(), ev.zero) &&
      ev.isGreaterEq(ev.fromType(dim), target.max()), "MultiLabelMarginCriterion: " +
      s"target out of range, target min should be greater than or equal to zero, but get " +
      s"${target.min()}, target max should be less than or equal to $dim, but get ${target.max()}")

    require(ev.isGreaterEq(isTarget.min(), ev.zero) &&
      ev.isGreaterEq(ev.fromType(dim), isTarget.max()), "MultiLabelMarginCriterion: " +
      "target out of range")

    val _target = target.contiguous()
    val _input = input.contiguous()
    val _isTarget = isTarget.contiguous()

    val input_data = _input.storage().array()
    val target_data = _target.storage().array()
    val isTarget_data = _isTarget.storage().array()

    val input_offset = _input.storageOffset() - 1
    val target_offset = _target.storageOffset() - 1
    val isTarget_offset = _isTarget.storageOffset() - 1

    val g = ev.fromType(if (sizeAverage)  1.0/(nframe*dim) else 1.0/(dim))

    gradInput.resizeAs(input).zero()
    val gradInput_data = gradInput.storage().array()

    var t = 0
    var n = 0
    while (t < nframe) {
      var dt = 0
      while (dt < dim) {
        val target_idx = ev.toType[Int](target_data(n + dt + target_offset)) - 1
        if (target_idx >= 0) {
          val input_target = input_data(n + target_idx + input_offset)
          var d = 0
          while (d < dim) {
            if (isTarget_data(n + d + isTarget_offset) == 0) {
              val z = ev.plus(ev.minus(ev.fromType(1), input_target),
                input_data(d + n + input_offset))
              if (ev.isGreater(z, ev.fromType(0))) {
                gradInput_data(target_idx + n) = ev.minus(gradInput_data(target_idx + n), g)
                gradInput_data(d + n) = ev.plus(gradInput_data(d + n), g)
              }
            }
            d += 1
          }
          dt += 1
        } else {
          dt = dim
        }
      }
      n += dim
      t += 1
    }
    gradInput
  }



  override def toString(): String = {
    s"nn.MultiLabelMarginCriterion($sizeAverage)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MultiLabelMarginCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MultiLabelMarginCriterion[T] =>
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

object MultiLabelMarginCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : MultiLabelMarginCriterion[T] = {
    new MultiLabelMarginCriterion[T](sizeAverage)
  }
}
