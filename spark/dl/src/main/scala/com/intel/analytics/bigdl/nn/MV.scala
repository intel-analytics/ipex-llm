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
 * It is a module to perform matrix vector multiplication on two mini-batch inputs,
 * producing a mini-batch.
 *
 * @param trans whether make matrix transpose before multiplication
 */

@SerialVersionUID(- 555327285289166316L)
class MV[T: ClassTag](val trans: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {
  gradInput = T(Tensor[T], Tensor[T]())

  private def checkInputFormat(input: Table): (Tensor[T], Tensor[T]) = {
    require(input.length() == 2 && input(1).isInstanceOf[Tensor[T]] &&
      input(2).isInstanceOf[Tensor[T]], "Input must be two tensors")
    val m: Tensor[T] = input(1)
    val v: Tensor[T] = input(2)
    require(m.dim() == 2 || m.dim() == 3, "input matrix must be 2D or 3D" +
      s"input dim ${m.dim()}")
    require(v.dim() == 1 || v.dim() == 2, "input vector must be 1D or 2D" +
      s"input dim ${v.dim()}")

    (m, v)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    var (m, v) = checkInputFormat(input)

    if (m.dim() == 2) {
      require(v.dim() == 1, "vector must be 1D" +
        s"dim ${v.dim()}")

      if (trans) {
        m = m.transpose(1, 2)
      }
      require(m.size(2) == v.size(1), "matrix row count and vector length do not match" +
        s"matrix row count ${m.size(2)}" +
        s"vector length ${v.size(1)}")

      output.resize(m.size(1)).zero()
      output.mv(m, v)
    } else {
      require(v.dim() == 2, "vector must be 2D (batch dimension)" +
        s"dimension ${v.dim()}" )
      require(m.size(1) == v.size(1), "inputs must contain the same number of minibatches" +
        s"The numbers are ${m.size(1)} and ${v.size(1)}")

      if (trans) {
        m = m.transpose(2, 3)
      }
      require(m.size(3) == v.size(2), "matrix row count and vector length do not match" +
        s"matrix row count ${m.size(3)}" +
        s"vector length ${v.size(2)}")

      output.resize(m.size(1), m.size(2), 1).zero()
      output.bmm(m, v.view(v.size(1), v.size(2), 1)).resize(m.size(1), m.size(2))
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val (m, v) = checkInputFormat(input)

    gradInput[Tensor[T]](1).resizeAs(m).zero()
    gradInput[Tensor[T]](2).resizeAs(v).zero()

    require(gradOutput.dim() == 1 || gradOutput.dim() == 2,
      "arguments must be a 1D or 2D Tensor" +
        s"arguments dim ${gradOutput.dim()}")

    if (gradOutput.dim() == 2) {
      require(m.dim() == 3, "matrix must must be 3D (batched)" +
        s"matrix dim ${m.dim()}")
      require(v.dim() == 2, "vector must be 2D (batched)" +
        s"vector dim ${v.dim()}")
      val bdim = m.size(1)
      val odim = m.size(2)
      val idim = m.size(3)

      if (trans) {
        gradInput[Tensor[T]](1).bmm(v.view(bdim, odim, 1), gradOutput.view(bdim, 1, idim))
        gradInput[Tensor[T]](2).view(bdim, odim, 1).bmm(m, gradOutput.view(bdim, odim, 1))
      } else {
        gradInput[Tensor[T]](1).bmm(gradOutput.view(bdim, odim, 1), v.view(bdim, 1, idim))
        gradInput[Tensor[T]](2).view(bdim, odim, 1).bmm(m.transpose(2, 3), gradOutput.view(bdim,
          odim, 1))
      }
    } else {
      require(m.dim() == 2, "matrix must be 2D" +
        s"matrix dimension ${m.dim()}")
      require(v.dim() == 1, "vector must be 1D" +
        s"vector dimension ${v.dim()}")

      if (trans) {
        gradInput[Tensor[T]](1).set(v.clone().resize(v.size(1), 1) *
          gradOutput.clone().resize(1, gradOutput.size(1)))
        gradInput[Tensor[T]](2).set(m * gradOutput)
      } else {
        gradInput[Tensor[T]](1).set(gradOutput.clone().resize(gradOutput.size(1), 1) *
          v.clone().resize(1, v.size(1)))
        gradInput[Tensor[T]](2).set(m.t() * gradOutput)
      }
    }

    gradInput
  }

  override def toString: String = s"MV()"

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MV[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MV[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        trans == that.trans
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), trans)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def clearState(): MV.this.type = {
    super.clearState()

    gradInput[Tensor[T]](1).set()
    gradInput[Tensor[T]](2).set()

    this
  }
}

object MV {
  def apply[@specialized(Float, Double) T: ClassTag](
      trans: Boolean = false)(implicit ev: TensorNumeric[T]) : MV[T] = {
    new MV[T](trans)
  }
}
