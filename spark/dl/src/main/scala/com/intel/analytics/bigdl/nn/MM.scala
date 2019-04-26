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
 * Module to perform matrix multiplication on two mini-batch inputs,
 * producing a mini-batch.
 *
 * @param transA specifying whether or not transpose the first input matrix
 * @param transB specifying whether or not transpose the second input matrix
 */

@SerialVersionUID(8315388141765786231L)
class MM[T: ClassTag](
  val transA: Boolean = false,
  val transB: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {
  gradInput = T(Tensor[T], Tensor[T]())

  private def checkInputFormat(input: Table): (Tensor[T], Tensor[T]) = {
    require(input.length() == 2 && input(1).isInstanceOf[Tensor[T]] &&
      input(2).isInstanceOf[Tensor[T]], "Input must be two tensors")
    val m1: Tensor[T] = input(1)
    val m2: Tensor[T] = input(2)
    require(m1.dim() == 2 || m1.dim() == 3 || m1.dim() == 4, "input matrix must be 2D or 3D or 4D" +
      s"input dim ${m1.dim()}")
    require(m2.dim() == 2 || m2.dim() == 3 || m2.dim() == 4, "input matrix must be 2D or 3D or 4D" +
      s"input dim ${m2.dim()}")

    (m1, m2)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    var (ma, mb) = checkInputFormat(input)

    if (ma.dim() == 2) {
      require(mb.dim() == 2, "second input tensor must be 2D" +
        s"second input dim ${mb.dim()}")

      if (transA) {
        ma = ma.t()
      }
      if (transB) {
        mb = mb.t()
      }
      require(ma.size(2) == mb.size(1), "matrix sizes do not match" +
        s"The sizes are ${ma.size(2)} and ${mb.size(1)}")

      output.resize(ma.size(1), mb.size(2))
      output.mm(ma, mb)
    } else {
      require(ma.dim() == mb.dim(), s"input tensors should be with same dimension," +
        s"but get ${ma.dim()} ${mb.dim()}")
      require(mb.dim() == 3 || mb.dim() == 4, "input tensor must be 3D or 4D, but get " +
        s"input dim ${mb.dim()}")

      val dimNum = ma.dim()
      val batchSizeX = ma.size().slice(0, dimNum - 2).product
      val batchSizeY = mb.size().slice(0, dimNum - 2).product
      require(batchSizeX == batchSizeY, "inputs must contain the same number of minibatches" +
        s"The minibatches of each are ${batchSizeX} and ${batchSizeY}")

      var reshapedX = ma.view(Array(batchSizeX, ma.size(dimNum - 1), ma.size(dimNum)))
      var reshapedY = mb.view(Array(batchSizeX, mb.size(dimNum - 1), mb.size(dimNum)))

      if (transA) {
        reshapedX = reshapedX.transpose(2, 3)
      }
      if (transB) {
        reshapedY = reshapedY.transpose(2, 3)
      }
      require(reshapedX.size(3) == reshapedY.size(2), "matrix sizes do not match" +
        s"the matrix sizes are ${reshapedX.size(3)} and ${reshapedY.size(2)}")

      output.resize(batchSizeX, reshapedX.size(2), reshapedY.size(3)).zero()
      output.bmm(reshapedX, reshapedY)
      val outputSize = ma.size().slice(0, dimNum - 2) ++ Array(reshapedX.size(2), reshapedY.size(3))
      output.resize(outputSize)
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val (ma, mb) = checkInputFormat(input)

    require(gradOutput.dim() == 2 || gradOutput.dim() == 3 || gradOutput.dim() == 4,
      "arguments must be a 2D or 3D or 4D Tensor" +
        s"arguments dim ${gradOutput.dim()}")


    val (hDim, wDim, f): (Int, Int, Tensor[T] => Tensor[T] => Tensor[T] => Tensor[T]) =
      if (gradOutput.dim() == 2) {
        require(ma.dim() == 2, "first input tensor must be 2D" +
          s"first input dim ${ma.dim()}")
        require(mb.dim() == 2, "second input tensor must be 2D" +
          s"second input dim ${mb.dim()}")

        (1, 2, t => m1 => m2 => t.mm(m1, m2))
      } else if (gradOutput.dim() == 3) {
        require(ma.dim() == 3, "first input tensor must be 3D" +
          s"first input dim ${ma.dim()}")
        require(mb.dim() == 3, "second input tensor must be 3D" +
          s"second input dim ${mb.dim()}")

        (2, 3, t => m1 => m2 => t.baddbmm(ev.fromType[Float](0.0f), ev.fromType[Float](1.0f),
          m1, m2))
      } else {
        require(ma.dim() == 4, "first input tensor must be 4D" +
          s"first input dim ${ma.dim()}")
        require(mb.dim() == 4, "second input tensor must be 4D" +
          s"second input dim ${mb.dim()}")

        (2, 3, t => m1 => m2 => t.bmm(m1, m2))
      }

    val dimNum = ma.dim()
    val batchSize = mb.size().slice(0, dimNum - 2).product
    val batchSizeGrad = gradOutput.size().slice(0, dimNum - 2).product

    var reshapedX = if (ma.dim() == 4) {
      ma.view(Array(batchSize, ma.size(dimNum - 1), ma.size(dimNum)))
    } else ma
    var reshapedY = if (mb.dim() == 4) {
      mb.view(Array(batchSize, mb.size(dimNum - 1), mb.size(dimNum)))
    } else mb
    val reshapeGradOutput = if (gradOutput.dim() == 4) {
      gradOutput.contiguous().view(batchSizeGrad,
        gradOutput.size(dimNum - 1), gradOutput.size(dimNum))
    } else gradOutput.contiguous()

    gradInput[Tensor[T]](1).resizeAs(reshapedX).zero()
    gradInput[Tensor[T]](2).resizeAs(reshapedY).zero()

    if (transA == transB) {
      reshapedX = reshapedX.transpose(hDim, wDim)
      reshapedY = reshapedY.transpose(hDim, wDim)
    }

    if (transA) {
      f (gradInput[Tensor[T]](1)) (reshapedY) (reshapeGradOutput.clone().transpose(hDim, wDim))
    } else {
      f (gradInput[Tensor[T]](1)) (reshapeGradOutput) (reshapedY)
    }

    if (transB) {
      f (gradInput[Tensor[T]](2)) (reshapeGradOutput.clone().transpose(hDim, wDim)) (reshapedX)
    } else {
      f (gradInput[Tensor[T]](2)) (reshapedX) (reshapeGradOutput)
    }

    gradInput[Tensor[T]](1).resizeAs(ma)
    gradInput[Tensor[T]](2).resizeAs(mb)
    gradInput
  }

  override def toString: String = s"MM()"

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MM[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MM[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        transA == that.transA &&
        transB == that.transB
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), transA, transB)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def clearState(): MM.this.type = {
    super.clearState()

    gradInput[Tensor[T]](1).set()
    gradInput[Tensor[T]](2).set()

    this
  }
}

object MM {
  def apply[@specialized(Float, Double) T: ClassTag](
      transA: Boolean = false,
      transB: Boolean = false)(implicit ev: TensorNumeric[T]) : MM[T] = {
    new MM[T](transA, transB)
  }
}
