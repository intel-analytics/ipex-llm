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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.Padding._

import scala.reflect.ClassTag

/**
 * This module adds paddings to the input tensor.
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using nInputDims.
 *
 * @param paddings how to pad the input
 * @param nInputDim specify the number of dimensions that this module will work on
 *                  If it is not equal to the dimension number of input tensors, the first dimension
 *                  would be considered as batch size
 * @param value padding value
 */
@SerialVersionUID(- 3401298839313169602L)
class Padding[T: ClassTag](
  val paddings: Array[PaddingInfo],
  val nInputDim: Int,
  val value: Double = 0.0)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    // Expand the output tensor
    val batch = (nInputDim != input.nDimension())
    inputs(0) = input
    var i = 0
    while(i < paddings.length) {
      inputs(i + 1) = paddingOneDimForward(inputs(i), paddings(i), batch)
      i += 1
    }
    output = inputs(i)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val batch = (nInputDim != input.nDimension())
    // Copy the tensor content
    var gradOutputDim = gradOutput
    var i = 0
    while(i < paddings.length) {
      gradOutputDim = paddingOneDimBackward(inputs(paddings.length - i - 1),
        gradOutputDim, paddings(paddings.length - i - 1), batch)
      i += 1
    }

    gradInput = gradOutputDim
    gradInput
  }

  private def paddingOneDimBackward(input: Tensor[T], gradOutput: Tensor[T],
                                    padInfo: PaddingInfo, batch: Boolean): Tensor[T] = {
    val gradInput = Tensor[T]().resizeAs(input)
    val dim = if (batch) padInfo.dim + 1 else padInfo.dim
    if(padInfo.leftStartIndex == 1 && padInfo.rightStartIndex == 1) {
      gradInput.narrow(dim, 1, input.size(dim))
        .copy(gradOutput.narrow(dim, padInfo.leftPadding + 1, input.size(dim)))
    } else {
      val central = input.size(dim) - (padInfo.leftStartIndex - 1) - (padInfo.rightStartIndex - 1)
      if(padInfo.leftStartIndex != 1) {
        gradInput.narrow(dim, 1, padInfo.leftStartIndex - 1)
          .copy(gradOutput.narrow(dim, 1, padInfo.leftStartIndex - 1))
      }
      if (central != 0) {
        gradInput.narrow(dim, padInfo.leftStartIndex, central)
          .copy(gradOutput.narrow(dim, padInfo.leftStartIndex + padInfo.leftPadding, central))
      }
      if(padInfo.rightStartIndex != 1) {
        gradInput.narrow(dim, input.size(dim) - padInfo.rightStartIndex + 2,
          padInfo.rightStartIndex - 1).copy(gradOutput.narrow(dim,
          padInfo.leftStartIndex + central + padInfo.rightPadding, padInfo.rightStartIndex - 1))
      }
    }
    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}((${paddings.mkString(",")}), $nInputDim, $value)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Padding[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Padding[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        paddings == that.paddings &&
        nInputDim == that.nInputDim &&
        value == that.value
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), paddings, nInputDim, value)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  private val inputs = new Array[Tensor[T]](paddings.length + 1)
  inputs.foreach(_ => Tensor[T]())

  private def paddingOneDimForward(input: Tensor[T], padInfo: PaddingInfo,
                                   batch: Boolean): Tensor[T] = {
    // Expand the output tensor
    val outputSize = input.size()
    val dim = if (batch) padInfo.dim + 1 else padInfo.dim
    outputSize(dim - 1) += padInfo.leftPadding + padInfo.rightPadding
    val output = Tensor(outputSize).fill(ev.fromType(value))

    // Copy the tensor content
    if(padInfo.leftStartIndex == 1 && padInfo.rightStartIndex == 1) {
      output.narrow(dim, padInfo.leftPadding + 1, input.size(dim))
        .copy(input.narrow(dim, 1, input.size(dim)))
    } else {
      val central = input.size(dim) - (padInfo.leftStartIndex - 1) - (padInfo.rightStartIndex - 1)
      if (padInfo.leftStartIndex != 1) {
        output.narrow(dim, 1, padInfo.leftStartIndex - 1)
          .copy(input.narrow(dim, 1, padInfo.leftStartIndex - 1))
      }
      if (central != 0) {
        output.narrow(dim, padInfo.leftStartIndex + padInfo.leftPadding, central)
          .copy(input.narrow(dim, padInfo.leftStartIndex, central))
      }
      if (padInfo.rightStartIndex != 1) {
        output.narrow(dim,
          outputSize(dim - 1) - padInfo.rightStartIndex + 2,
          padInfo.rightStartIndex - 1).copy(input.narrow(dim,
          input.size(dim) - padInfo.rightStartIndex + 2, padInfo.rightStartIndex - 1))
      }
    }
    output
  }
}

object Padding {
  /**
   * Padding information. It define which dimensions to add padding
   * @param dim the index of the dimension to add padding, start from 1
   * @param leftPadding left padding length
   * @param leftStartIndex left padding start index
   * @param rightPadding right padding length
   * @param rightStartIndex right padding start index
   */
  case class PaddingInfo(
    dim: Int,
    leftPadding: Int,
    leftStartIndex: Int,
    rightPadding: Int,
    rightStartIndex: Int
  )

  def apply[T: ClassTag](
      dim: Int,
      pad: Int,
      nInputDim: Int,
      value: Double = 0.0,
      nIndex: Int = 1)(implicit ev: TensorNumeric[T]): Padding[T] = {
    if (pad > 0) {
      new Padding[T](Array(PaddingInfo(dim, 0, 1, pad, nIndex)), nInputDim, value)
    } else {
      new Padding[T](Array(PaddingInfo(dim, -pad, nIndex, 0, 1)), nInputDim, value)
    }
  }

  def apply[T: ClassTag](
    paddings: Array[PaddingInfo],
    nInputDim: Int,
    value: Double)(implicit ev: TensorNumeric[T]) : Padding[T] = {
    new Padding[T](paddings, nInputDim, value)
  }
}
