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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * It is a table module which takes a table of Tensors as input and
 * outputs a Tensor by joining them together along the dimension `dimension`.
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using `nInputDims`.
 *
 * @param dimension to be join in this dimension
 * @param nInputDims specify the number of dimensions that this module will receive
 *                   If it is more than the dimension of input tensors, the first dimension
 *                   would be considered as batch size
 */

@SerialVersionUID(- 8435694717504118735L)
class JoinTable[T: ClassTag] (
  val dimension: Int,
  val nInputDims: Int
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {

  private def getPositiveDimension(input: Table): Int = {
    var nDim = this.dimension
    val firstInput: Tensor[T] = input(1)

    if (nDim < 0) {
      nDim = firstInput.dim() + nDim + 1
    } else if (nInputDims > 0 && firstInput.dim() == (nInputDims + 1)) {
      nDim += 1
    }
    require(firstInput.dim() >= dimension, "dimension exceeds input dimensions")
    nDim
  }

  override def updateOutput(input: Table): Tensor[T] = {
    val dimension = getPositiveDimension(input)
    var size: Array[Int] = null

    var i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      if (i == 1) {
        size = currentOutput.size()
      } else {
        size(dimension - 1) += currentOutput.size(dimension)
      }
      i += 1
    }
    output.resize(size)

    var offset = 1
    i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      output.narrow(dimension, offset, currentOutput.size(dimension))
        .copy(currentOutput)
      offset += currentOutput.size(dimension)
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val dimension = getPositiveDimension(input)

    var i = 1
    while (i <= input.length()) {
      if (!gradInput.contains(i)) {
        gradInput(i) = Tensor()
      }
      gradInput[Tensor[T]](i).resizeAs(input(i))
      i += 1
    }

    var offset = 1
    i = 1
    while (i <= input.length()) {
      val currentOutput: Tensor[T] = input(i)
      val currentGradInput = gradOutput
        .narrow(dimension, offset, currentOutput.size(dimension))
      gradInput[Tensor[T]](i)copy(currentGradInput)
      offset += currentOutput.size(dimension)
      i += 1
    }
    gradInput
  }

  override def toString: String = s"nn.JoinTable"


  override def canEqual(other: Any): Boolean = other.isInstanceOf[JoinTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: JoinTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dimension == that.dimension &&
        nInputDims == that.nInputDims
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), dimension, nInputDims)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object JoinTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int,
      nInputDims: Int)(implicit ev: TensorNumeric[T]) : JoinTable[T] = {
    new JoinTable[T](dimension, nInputDims)
  }
}
