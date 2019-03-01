/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import java.security.InvalidParameterException

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Module to perform matrix multiplication on two mini-batch inputs,
 * producing a mini-batch. Input with different batchSize are supported.
 *
 * @param transA specifying whether or not transpose the first input matrix
 * @param transB specifying whether or not transpose the second input matrix
 */

@SerialVersionUID(8315388141765786231L)
private[zoo] class InternalMM[T: ClassTag](
    val transA: Boolean = false,
    val transB: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {
  gradInput = T(Tensor[T], Tensor[T]())

  private var expandLayer: AbstractModule[Tensor[T], Tensor[T], T] = null

  private def checkInputFormat(input: Table): (Tensor[T], Tensor[T]) = {
    require(input.length() == 2 && input(1).isInstanceOf[Tensor[T]] &&
      input(2).isInstanceOf[Tensor[T]], "Input must be two tensors")
    val m1: Tensor[T] = input(1)
    val m2: Tensor[T] = input(2)
    require(m1.dim() == 2 || m1.dim() == 3, "input matrix must be 2D or 3D" +
      s" input dim ${m1.dim()}")
    require(m2.dim() == 2 || m2.dim() == 3, "input matrix must be 2D or 3D" +
      s" input dim ${m2.dim()}")

    (m1, m2)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    var (ma, mb) = checkInputFormat(input)
    output.set()

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
      require(mb.dim() == 3, "second input tensor must be 3D" +
        s"second input dim ${mb.dim()}")
      if (transA) {
        ma = ma.transpose(2, 3)
      }
      if (transB) {
        mb = mb.transpose(2, 3)
      }
      require(ma.size(3) == mb.size(2), "matrix sizes do not match" +
        s"the matrix sizes are ${ma.size(3)} and ${mb.size(2)}")

      // with different batch dim
      if (ma.size(1) != mb.size(1)) {
        val newTensors = expandTensor(ma, mb)
        ma = newTensors._1
        mb = newTensors._2
      }
      output.resize(ma.size(1), ma.size(2), mb.size(3))
      output.bmm(ma, mb)
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput[Tensor[T]](1).set()
    gradInput[Tensor[T]](2).set()

    var (ma, mb) = checkInputFormat(input)

    if (ma.dim() > 2 && ma.size(1) != mb.size(1)) {
      require(mb.dim() == 3, "second input tensor must be 3D" +
        s"second input dim ${mb.dim()}")
      // with different batch dim
      val newTensors = expandTensor(ma, mb)
      ma = newTensors._1
      mb = newTensors._2
    }

    gradInput[Tensor[T]](1).resizeAs(ma)
    gradInput[Tensor[T]](2).resizeAs(mb)

    require(gradOutput.dim() == 2 || gradOutput.dim() == 3,
      "arguments must be a 2D or 3D Tensor" +
        s"arguments dim ${gradOutput.dim()}")


    val (hDim, wDim, f): (Int, Int, Tensor[T] => Tensor[T] => Tensor[T] => Tensor[T]) =
      if (gradOutput.dim() == 2) {
        require(ma.dim() == 2, "first input tensor must be 2D" +
          s"first input dim ${ma.dim()}")
        require(mb.dim() == 2, "second input tensor must be 2D" +
          s"second input dim ${mb.dim()}")

        (1, 2, t => m1 => m2 => t.mm(m1, m2))
      } else {
        require(ma.dim() == 3, "first input tensor must be 3D" +
          s"first input dim ${ma.dim()}")
        require(mb.dim() == 3, "second input tensor must be 3D" +
          s"second input dim ${mb.dim()}")

        (2, 3, t => m1 => m2 => t.bmm(m1, m2))
      }


    if (transA == transB) {
      ma = ma.transpose(hDim, wDim)
      mb = mb.transpose(hDim, wDim)
    }

    if (transA) {
      f (gradInput[Tensor[T]](1)) (mb) (gradOutput.clone().transpose(hDim, wDim))
    } else {
      f (gradInput[Tensor[T]](1)) (gradOutput) (mb)
    }

    if (transB) {
      f (gradInput[Tensor[T]](2)) (gradOutput.clone().transpose(hDim, wDim)) (ma)
    } else {
      f (gradInput[Tensor[T]](2)) (ma) (gradOutput)
    }

    // with different batch dim
    if (ma.dim() > 2 && ma.size(1) != mb.size(1)) {
      require(mb.dim() == 3, "second input tensor must be 3D" +
        s"second input dim ${mb.dim()}")
      if (ma.size(1) == 1) {
        gradInput(1) = expandLayer.backward(ma, gradInput[Tensor[T]](1)).toTensor
      } else if (mb.size(1) == 1) {
        gradInput(2) = expandLayer.backward(mb, gradInput[Tensor[T]](2)).toTensor
      } else {
        throw new InvalidParameterException("inputs must contain the same number of" +
          "minibatches. The minibatces of each are ${ma.size(1)} and ${mb.size(1)}\"")
      }
    }
    gradInput
  }

  override def toString: String = s"MM()"

  override def canEqual(other: Any): Boolean = other.isInstanceOf[InternalMM[T]]

  override def equals(other: Any): Boolean = other match {
    case that: InternalMM[T] =>
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

  override def clearState(): InternalMM.this.type = {
    super.clearState()

    gradInput[Tensor[T]](1).set()
    gradInput[Tensor[T]](2).set()

    this
  }

  private def expandTensor(ma: Tensor[T], mb: Tensor[T]): (Tensor[T], Tensor[T]) = {
    val (newA, newB) = if (ma.size(1) == 1) {
      expandLayer = InternalExpand(mb.size())
      (expandLayer.forward(ma), mb)
    } else if (mb.size(1) == 1) {
      expandLayer = InternalExpand(ma.size())
      (ma, expandLayer.forward(mb))
    } else {
      throw new InvalidParameterException("inputs must contain the same number of" +
        "minibatches. The minibatces of each are ${ma.size(1)} and ${mb.size(1)}\"")
    }
    (newA, newB)
  }
}

private[zoo] object InternalMM {
  def apply[@specialized(Float, Double) T: ClassTag](
      transA: Boolean = false,
      transB: Boolean = false)(implicit ev: TensorNumeric[T]) : InternalMM[T] = {
    new InternalMM[T](transA, transB)
  }
}
