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
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Creates a module that takes a Tensor as input and
 * outputs several tables, splitting the Tensor along
 * the specified dimension `dimension`. Please note the dimension starts from 1.
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user needs to specify the number of dimensions of each sample tensor in a
 * batch using `nInputDims`.
 *
 * @param dimension to be split along this dimension
 * @param nInputDims specify the number of dimensions that this module will receive
 *                   If it is more than the dimension of input tensors, the first dimension
 *                   would be considered as batch size
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(- 4318640284973082779L)
class SplitTable[T: ClassTag](
  var dimension: Int,
  var nInputDims: Int = -1,
  var keepDim: Boolean = false,
  var contiguousOutput: Boolean = false
)(implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Table, T]{

  private def getPositiveDimension(input: Tensor[T]): Int = {
    if (dimension < 0) {
      input.dim() + dimension + 1
    } else if (dimension != Int.MaxValue && input.dim() == nInputDims + 1) {
      dimension + 1
    } else {
      dimension
    }
  }

  override def updateOutput(input: Tensor[T]): Table = {
    val dim = getPositiveDimension(input)
    val slices = input.size(dim)

    val currentOutput = T()
    var i = 1
    while (i <= slices) {
      val t = input.select(dim, i)
      if (keepDim) t.addSingletonDimension(t, dim)
      currentOutput.insert(if (contiguousOutput) t.contiguous() else t)
      i += 1
    }
    output = currentOutput

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    val dim = getPositiveDimension(input)
    val slices = input.size(dim)

    gradInput.resizeAs(input)

    var i = 1
    while (i <= slices) {
      val currentGradInput: Tensor[T] = gradOutput(i)
      gradInput.select(dim, i).copy(currentGradInput)
      i += 1
    }

    gradInput
  }


  override def canEqual(other: Any): Boolean = other.isInstanceOf[SplitTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: SplitTable[_] =>
      super.equals(that) &&
        (that canEqual this) &&
        dimension == that.dimension &&
        nInputDims == that.nInputDims
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), dimension, nInputDims)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }


  override def toString: String = s"SplitTable($dimension, $nInputDims)"
}

object SplitTable {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimension: Int,
    nInputDims: Int = -1,
    keepDim: Boolean = false,
    contiguousOutput: Boolean = false
  )(implicit ev: TensorNumeric[T]) : SplitTable[T] = {
    new SplitTable[T](dimension, nInputDims, keepDim, contiguousOutput)
  }
}
