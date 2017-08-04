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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Applies a min operation over dimension `dim`.
 *
 * @param dim min along this dimension
 * @param numInputDims Optional. If in a batch model, set to the inputDims.
 */

@SerialVersionUID(8958076163182151950L)
class Min[T: ClassTag](
  var dim : Int = 1,
  var numInputDims: Int = Int.MinValue
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T]  {

  private val values: Tensor[T] = Tensor[T]()
  private val indices: Tensor[T] = Tensor[T]()

  def setNumInputDims(numInputDims: Int): Unit = {
    this.numInputDims = numInputDims
  }

  private def getPositiveDimension(input: Tensor[T]): Int = {
    if (dim < 0) {
      input.dim() + dim + 1
    } else if (numInputDims != Int.MinValue && input.dim() == numInputDims + 1) {
      dim + 1
    } else {
      dim
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dimension = getPositiveDimension(input)
    input.min(values, indices, dimension)
    if (input.dim() > 1) {
      output.set(values.select(dimension, 1))
    } else {
      output.set(values)
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dimension = getPositiveDimension(input)
    val gradOutputView = if (input.dim() > 1) {
      Tensor[T]().addSingletonDimension(gradOutput, dimension)
    } else {
      gradOutput
    }
    gradInput.resizeAs(input).zero().scatter(dimension, indices, gradOutputView)

    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($dim${if (numInputDims != Int.MinValue) ", " + numInputDims else ""})"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Min[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Min[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dim == that.dim &&
        numInputDims == that.numInputDims
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), dim, numInputDims)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def clearState() : this.type = {
    super.clearState()
    values.set()
    indices.set()
    this
  }
}

object Min {
  def apply[@specialized(Float, Double) T: ClassTag](
      dim : Int = 1,
      numInputDims: Int = Int.MinValue)(implicit ev: TensorNumeric[T]) : Min[T] = {
    new Min[T](dim, numInputDims)
  }
}
