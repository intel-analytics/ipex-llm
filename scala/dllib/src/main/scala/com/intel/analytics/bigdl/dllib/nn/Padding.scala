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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.reflect.ClassTag

/**
 * This module adds pad units of padding to dimension dim of the input. If pad is negative,
 * padding is added to the left, otherwise, it is added to the right of the dimension.
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using nInputDims.
 *
 * @param dim the dimension to be applied padding operation
 * @param pad num of the pad units
 * @param nInputDim specify the number of dimensions that this module will receive
 *                  If it is more than the dimension of input tensors, the first dimension
 *                  would be considered as batch size
 * @param value padding value
 */
@SerialVersionUID(- 3401298839313169602L)
class Padding[T: ClassTag](
  val dim: Int,
  val pad: Int,
  val nInputDim: Int,
  val value: Double = 0.0,
  val nIndex: Int = 1)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var outputSize = Storage[Int]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    outputSize.resize(input.dim()).copy(Storage(input.size()))
    val dim = if (input.dim() != nInputDim) this.dim + 1 else this.dim

    outputSize(dim - 1) += math.abs(this.pad)
    output.resize(outputSize.array()).fill(ev.fromType(value))

    val index = if (this.pad > 0) input.size(dim) - nIndex + 2 else nIndex
    val pad = if (this.pad > 0) this.pad else -this.pad

    if (index == 1) {
      output.narrow(dim, 1 + pad, input.size(dim)).copy(input)
    } else if (index == (input.size(dim) + 1)) {
      output.narrow(dim, 1, input.size(dim)).copy(input)
    } else {
      output.narrow(dim, 1, index - 1).copy(input.narrow(dim, 1, index - 1))
      output.narrow(dim, index + pad, input.size(dim) - (index - 1)).
        copy(input.narrow(dim, index, input.size(dim) - (index - 1)))
    }
    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    val dim = if (input.dim() != nInputDim) this.dim + 1 else this.dim
    val index = if (this.pad > 0) input.size(dim) - nIndex + 2 else nIndex
    val pad = if (this.pad > 0) this.pad else -this.pad

    if (index == 1) {
      gradInput.copy(gradOutput.narrow(dim, 1 + pad, input.size(dim)))
    } else if (index == input.size(dim) + 1) {
      gradInput.copy(gradOutput.narrow(dim, 1, input.size(dim)))
    } else {
      gradInput.narrow(dim, 1, index - 1).
        copy(gradOutput.narrow(dim, 1, index - 1))
      gradInput.narrow(dim, index, input.size(dim) - (index - 1)).copy(
        gradOutput.narrow(dim, index + pad, input.size(dim) - (index - 1)))
    }
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($dim, $pad, $nInputDim, $value, $nIndex)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Padding[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Padding[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dim == that.dim &&
        pad == that.pad &&
        nInputDim == that.nInputDim &&
        value == that.value &&
        nIndex == that.nIndex
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), dim, pad, nInputDim, value, nIndex)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Padding{
  def apply[@specialized(Float, Double) T: ClassTag](
    dim: Int,
    pad: Int,
    nInputDim: Int,
    value: Double = 0.0,
    nIndex: Int = 1)(implicit ev: TensorNumeric[T]) : Padding[T] =
    new Padding[T](dim, pad, nInputDim, value, nIndex)
}
