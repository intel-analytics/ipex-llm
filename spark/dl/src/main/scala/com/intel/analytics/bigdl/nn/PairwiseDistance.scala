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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * It is a module that takes a table of two vectors as input and outputs
 * the distance between them using the p-norm.
 * The input given in `forward(input)` is a [[Table]] that contains two tensors which
 * must be either a vector (1D tensor) or matrix (2D tensor). If the input is a vector,
 * it must have the size of `inputSize`. If it is a matrix, then each row is assumed to be
 * an input sample of the given batch (the number of rows means the batch size and
 * the number of columns should be equal to the `inputSize`).
 *
 * @param norm the norm of distance
 */

@SerialVersionUID(- 4377017408738399127L)
class PairwiseDistance[T: ClassTag](
  val norm : Int = 2)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    output.resize(1)
    if (input[Tensor[T]](1).dim() == 1) {
      output.resize(1)
      output.setValue(1, input[Tensor[T]](1).dist(input(2), norm))
    } else if (input[Tensor[T]](1).dim() == 2) {
      val diff = Tensor[T]()
      diff
        .resizeAs(input(1))
        .zero()

      diff.add(input(1), ev.fromType[Int](-1), input(2))
      diff.abs()

      output.resize(input[Tensor[T]](1).size(1))
      output.zero()
      output.add(diff.pow(ev.fromType[Int](norm)).sum(2))
      output.pow(ev.divide(ev.fromType[Int](1), ev.fromType[Int](norm)))
    } else {
      require(requirement = false,
        "PairwiseDistance: " + ErrorInfo.constrainEachInputAsVectorOrBatch)
    }
    output
  }

  private def mathsign(x: T): T = {
    if (ev.equals(x, ev.zero)) {
      2 * RNG.uniform(0, 2) - 3
    }

    if (ev.isGreater(x, ev.zero)) {
      ev.one
    } else {
      ev.negative(ev.one)
    }
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    require(input[Tensor[T]](1).dim() <= 2,
      "PairwiseDistance : " + ErrorInfo.constrainEachInputAsVectorOrBatch)

    if (!gradInput.contains(1)) {
      gradInput.update(1, Tensor[T]())
    }

    if (!gradInput.contains(2)) {
      gradInput.update(2, Tensor[T]())
    }

    gradInput[Tensor[T]](1).resizeAs(input[Tensor[T]](1))
    gradInput[Tensor[T]](2).resizeAs(input[Tensor[T]](2))

    gradInput[Tensor[T]](1)
      .copy(input[Tensor[T]](1))
      .add(ev.negative(ev.one), input[Tensor[T]](2))

    if (norm == 1) {
      gradInput[Tensor[T]](1).apply1(mathsign)
    } else {
      if (norm > 2) {
        gradInput[Tensor[T]](1)
          .cmul(gradInput[Tensor[T]](1)
            .clone()
            .abs()
            .pow(ev.minus(ev.fromType[Int](norm), ev.fromType[Int](2))))
      }
    }

    if (input[Tensor[T]](1).dim() > 1) {
      val outExpand = Tensor[T]()
      outExpand
        .resize(output.size(1), 1)
        .copy(output)
        .add(ev.fromType[Double](1.0e-6))
        .pow(ev.negative(ev.fromType[Int](norm - 1)))

      gradInput[Tensor[T]](1).cmul(
        outExpand.expand(
          Array(gradInput[Tensor[T]](1).size(1), gradInput[Tensor[T]](1).size(2))))
    } else {
      gradInput[Tensor[T]](1).mul(
        ev.pow(
          ev.plus(output.apply(Array(1)), ev.fromType[Double](1e-6)), ev.fromType[Int](1 - norm)))
    }

    if (input[Tensor[T]](1).dim() == 1) {
      gradInput[Tensor[T]](1).mul(gradOutput(Array(1)))
    } else {
      val grad = Tensor[T]()
      val ones = Tensor[T]()

      grad
        .resizeAs(input[Tensor[T]](1))
        .zero()

      ones
        .resize(input[Tensor[T]](1).size(2))
        .fill(ev.one)

      grad
        .addr(gradOutput, ones)
      gradInput[Tensor[T]](1).cmul(grad)
    }

    gradInput[Tensor[T]](2)
      .zero()
      .add(ev.negative(ev.one), gradInput[Tensor[T]](1))
    gradInput
  }

  override def toString: String = {
    s"nn.PairwiseDistance"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[PairwiseDistance[T]]


  override def equals(other: Any): Boolean = other match {
    case that: PairwiseDistance[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        norm == that.norm
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), norm)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object PairwiseDistance {
  def apply[@specialized(Float, Double) T: ClassTag](
      norm : Int = 2)(implicit ev: TensorNumeric[T]) : PairwiseDistance[T] = {
    new PairwiseDistance[T](norm)
  }
}
