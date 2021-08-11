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
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Creates a criterion that measures the loss given an input x = {x1, x2},
  * a table of two Tensors, and a label y (1 or -1):
 *
 * @param margin
 */

@SerialVersionUID(- 1765228642089353823L)
class L1HingeEmbeddingCriterion[@specialized(Float, Double) T: ClassTag](val margin: Double = 1)
 (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Table, Tensor[T], T]{

  private def mathSign(t: T): T = {
    var res = 0
    if (ev.isGreater(t, ev.fromType(0))) {
      res = 1
    } else if (ev.isGreater(ev.fromType(0), t)) {
      res = -1
    } else {
      res = 2 * (Math.floor(RNG.uniform(0, 2)).toInt + 1) - 3
    }
    ev.fromType(res)
  }

  override def updateOutput(input: Table, target: Tensor[T]): T = {
    require(target.dim() == 1 && target.nElement() == 1,
      "L1HingeEmbeddingCriterion.updateOutput: " +
        "target should be vector with one element," +
        s" target shape [${target.dim()},${target.nElement()}]")
    val y = target.valueAt(1)
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)

    output = (input1 -input2).abs().sum()
    if (y == -1) {
      output = ev.max(ev.fromType(0), ev.minus(ev.fromType(margin), output))
    }
    output
  }

  override def updateGradInput(input: Table, target: Tensor[T]): Table = {
    require(target.dim() == 1 && target.nElement() == 1,
      s"L1HingeEmbeddingCriterion.updateOutput:" +
        " target should be vector with one element," +
        s" target shape [${target.dim()},${target.nElement()}]")
    val y = target.valueAt(1)
    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T])
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T])

    val gradInput1 = gradInput[Tensor[T]](1)
    val gradInput2 = gradInput[Tensor[T]](2)
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)

    gradInput1.resizeAs(input1).copy(input1)
    gradInput1.add(ev.fromType(-1), input2)
    gradInput2.resizeAs(input2)

    val dist = gradInput1.norm(1)
    gradInput1.apply1(mathSign)

    if (y == -1) {
      if (ev.isGreater(dist, ev.fromType(margin))) {
        gradInput1.zero()
      } else {
        gradInput1.mul(ev.fromType(-1))
      }
    }
    gradInput2.zero().add(ev.fromType(-1), gradInput1)
    gradInput
  }

  override def toString(): String = {
    s"nn.L1HingeEmbeddingCriterion ($margin)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[L1HingeEmbeddingCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: L1HingeEmbeddingCriterion[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        margin == that.margin
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), margin)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object L1HingeEmbeddingCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      margin: Double = 1)(implicit ev: TensorNumeric[T]) : L1HingeEmbeddingCriterion[T] = {
    new L1HingeEmbeddingCriterion[T](margin)
  }
}
