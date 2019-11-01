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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

/**
 * Insert singleton dim (i.e., dimension 1) at position array pos.
 * For an input with dim = input.dim(),
 * there are dim + 1 possible positions to insert the singleton dimension.
 * Dimension index are 1-based. 0 and negative pos correspond to unsqueeze() applied at
 * pos = pos + input.dim() + 1
 *
 * @param pos The array of position will insert singleton.
 * @param numInputDims Optional. If in a batch model, set to the inputDims.
 */

@SerialVersionUID(- 5180889605872472241L)
class Unsqueeze[T: ClassTag](
  val pos: Array[Int],
  var numInputDims: Int = Int.MinValue
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[_], Tensor[_], T]  {
  def this( pos: Int, numInputDims: Int )(implicit ev: TensorNumeric[T]) = {
    this(Array(pos), numInputDims)
  }

  def this( pos: Int )(implicit ev: TensorNumeric[T]) = {
    this(Array(pos))
  }

  def setNumInputDims(numInputDims: Int): Unit = {
    this.numInputDims = numInputDims
  }

  private def getActualPosition(input: Tensor[_]) : Array[Int] = {
    for (index <- 0 until pos.length) {
      // dimension index are 1-based
      pos(index) = if (pos(index) <= 0) {
        input.dim() + pos(index) + 1
      }
      else {
        pos(index)
      }
      // get valid dimension offset for batchMode (if any)
      val inputDim = input.dim() // data batch dim
      numInputDims = if (numInputDims != Int.MinValue) numInputDims else inputDim // feature map dim
      val offsetDim = inputDim - numInputDims
      require(offsetDim >= 0, "input feature map dim (numInputDims) must be <= input:dim()," +
        s" input feature map dim ${numInputDims}, inputdim ${inputDim}")
      // the actual position; clearer error message for batchMode (if any)
      val actualPos = pos(index) + offsetDim
      require(actualPos >= 1 && actualPos <= (inputDim + 1), s"Invalid position: ${pos(index)}. " +
        s"input:dim() is $input, input feature map dim (numInputDims) is $numInputDims.")
      pos(index) = actualPos
    }
    pos
  }

  override def updateOutput(input: Tensor[_]): Tensor[_] = {
    val actualPos = getActualPosition(input)
    if (input.getType() != output.getType()) {
      output = input.emptyInstance()
    }

    output.asInstanceOf[Tensor[NumericWildcard]]
    .addMultiDimension(input.asInstanceOf[Tensor[NumericWildcard]], actualPos)

    output
  }

  override def updateGradInput(input: Tensor[_], gradOutput: Tensor[_]): Tensor[_] = {
    require(input.nElement() == gradOutput.nElement(),
      "input and gradOutput should be of the same size" +
        s"input size ${input.nElement()} gradOutput size ${gradOutput.nElement()}")
    gradInput = gradOutput.view(input.size())
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($pos${if (numInputDims != Int.MinValue) ", " + numInputDims else ""})"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Unsqueeze[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Unsqueeze[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        pos == that.pos &&
        numInputDims == that.numInputDims
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), pos, numInputDims)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Unsqueeze {
  def apply[@specialized(Float, Double) T: ClassTag](
    pos: Int,
    numInputDims: Int = Int.MinValue)(implicit ev: TensorNumeric[T]) : Unsqueeze[T] = {
    new Unsqueeze[T](Array(pos), numInputDims)
  }

  def apply[T: ClassTag](
    posList: Array[Int],
    numInputDims: Int)(implicit ev: TensorNumeric[T]) : Unsqueeze[T] = {
    new Unsqueeze[T](posList, numInputDims)
  }

  def apply[T: ClassTag](
    posList: Array[Int])(implicit ev: TensorNumeric[T]) : Unsqueeze[T] = {
    new Unsqueeze[T](posList)
  }
}
