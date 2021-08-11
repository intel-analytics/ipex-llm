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
 * Creates a criterion that measures the loss given an input x = {x1, x2},
 * a table of two Tensors of size 1 (they contain only scalars), and a label y (1 or -1).
 * In batch mode, x is a table of two Tensors of size batchsize, and y is a Tensor of size
 * batchsize containing 1 or -1 for each corresponding pair of elements in the input Tensor.
 * If y == 1 then it assumed the first input should be ranked higher (have a larger value) than
 * the second input, and vice-versa for y == -1.
 *
 * @param margin
 */

@SerialVersionUID(4746239527786180108L)
class MarginRankingCriterion[@specialized(Float, Double) T: ClassTag]
(val margin: Double = 1.0, val sizeAverage: Boolean = true)
 (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Table, Table, T] {

  @transient
  var mask: Tensor[T] = null
  @transient
  var dist: Tensor[T] = null

  override def updateOutput(input: Table, y: Table): T = {
    // todo: number condition
    val target = y[Tensor[T]](1)
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)

    if (target.nElement() == 1) {
      val v1 = ev.minus(input1(Array(1)), input2(Array(1)))
      val v2 = ev.negative(target(Array(1)))
      output = ev.max(ev.fromType(0), ev.plus(ev.times(v1, v2), ev.fromType(margin)))
    } else {
      if (null == dist) dist = Tensor[T]()
      dist.resizeAs(input1).copy(input1)
      dist.add(ev.fromType(-1), input2).mul(ev.fromType(-1)).cmul(target)
      dist.add(ev.fromType(margin))
      dist.cmax(ev.fromType(0))
      output = dist.sum()
      if (sizeAverage) output = ev.divide(output, ev.fromType(target.size(1)))
    }
    output
  }

  override def updateGradInput(input: Table, y: Table): Table = {
    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T](1))
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T](1))
    // todo: number condition
    val target = y[Tensor[T]](1)
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)
    val gradInput1 = gradInput[Tensor[T]](1)
    val gradInput2 = gradInput[Tensor[T]](2)

    if (target.nElement() == 1) {
      val v1 = ev.minus(input1(Array(1)), input2(Array(1)))
      val v2 = target(Array(1))
      val dist = ev.toType[Double](v1) * ev.toType[Double](v2) * (-1) + margin
      if (dist < 0) {
        gradInput1.setValue(1, ev.fromType(0))
        gradInput2.setValue(1, ev.fromType(0))
      } else {
        gradInput1.setValue(1, ev.negative(v2))
        gradInput2.setValue(1, v2)
      }
    } else {
      if (null == mask) mask = Tensor[T]()
      if (null == dist) dist = Tensor[T]()
      dist.resizeAs(input1).copy(input1)
      dist.add(ev.fromType(-1), input[Tensor[T]](2))
      dist.mul(ev.fromType(-1)).cmul(target).add(ev.fromType(margin))

      mask.resizeAs(input1).copy(dist)
      mask.ge(dist, 0)
      gradInput1.resizeAs(dist).copy(mask).mul(ev.fromType(-1)).cmul(target)
      gradInput2.resizeAs(dist).copy(mask).cmul(target)

      if (sizeAverage) {
        gradInput1.div(ev.fromType(target.size(1)))
        gradInput2.div(ev.fromType(target.size(1)))
      }
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MarginRankingCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MarginRankingCriterion[T] =>
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

  override def toString(): String = {
    s"nn.MarginRankingCriterion($margin)"
  }
}

object MarginRankingCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      margin: Double = 1.0,
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : MarginRankingCriterion[T] = {
    new MarginRankingCriterion[T](margin, sizeAverage)
  }
}
