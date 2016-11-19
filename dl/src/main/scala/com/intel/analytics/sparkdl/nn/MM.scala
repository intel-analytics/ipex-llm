/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Module to perform matrix multiplication on two mini-batch inputs,
 * producing a mini-batch.
 * @param transA specifying whether or not transpose the first input matrix
 * @param transB specifying whether or not transpose the second input matrix
 */
class MM[T: ClassTag](
  val transA: Boolean = false,
  val transB: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {
  gradInput = T(Tensor[T], Tensor[T]())

  private def checkInputFormat(input: Table): (Tensor[T], Tensor[T]) = {
    require(input.getState().size == 2 && input(1).isInstanceOf[Tensor[T]] &&
      input(2).isInstanceOf[Tensor[T]], "Input must be two tensors")
    val m1: Tensor[T] = input(1)
    val m2: Tensor[T] = input(2)
    require(m1.dim() == 2 || m1.dim() == 3, "input matrix must be 2D or 3D")
    require(m2.dim() == 2 || m2.dim() == 3, "input matrix must be 2D or 3D")

    (m1, m2)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    var (ma, mb) = checkInputFormat(input)

    if (ma.dim() == 2) {
      require(mb.dim() == 2, "second input tensor must be 2D")

      if (transA) {
        ma = ma.t()
      }
      if (transB) {
        mb = mb.t()
      }
      require(ma.size(2) == mb.size(1), "matrix sizes do not match")

      output.resize(ma.size(1), mb.size(2))
      output.mm(ma, mb)
    } else {
      require(mb.dim() == 3, "second input tensor must be 3D")
      require(ma.size(1) == mb.size(1), "inputs must contain the same number of minibatches")

      if (transA) {
        ma = ma.transpose(2, 3)
      }
      if (transB) {
        mb = mb.transpose(2, 3)
      }
      require(ma.size(3) == mb.size(2), "matrix sizes do not match")

      output.resize(ma.size(1), ma.size(2), mb.size(3))
      output.bmm(ma, mb)
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var (ma, mb) = checkInputFormat(input)

    gradInput[Tensor[T]](1).resizeAs(ma)
    gradInput[Tensor[T]](2).resizeAs(mb)

    require(gradOutput.dim() == 2 || gradOutput.dim() == 3,
      "arguments must be a 2D or 3D Tensor")


    val (hDim, wDim, f): (Int, Int, Tensor[T] => Tensor[T] => Tensor[T] => Tensor[T]) =
      if (gradOutput.dim() == 2) {
        require(ma.dim() == 2, "first input tensor must be 2D")
        require(mb.dim() == 2, "second input tensor must be 2D")

        (1, 2, t => m1 => m2 => t.mm(m1, m2))
      } else {
        require(ma.dim() == 3, "first input tensor must be 3D")
        require(mb.dim() == 3, "second input tensor must be 3D")

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
}
