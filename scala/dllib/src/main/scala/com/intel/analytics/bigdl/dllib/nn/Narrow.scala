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

import scala.reflect.ClassTag

/**
 * Narrow is application of narrow operation in a module.
 * The module further supports a negative length in order to handle inputs with an unknown size.
 */
@SerialVersionUID(988790441682879293L)
class Narrow[T: ClassTag](dimension: Int, offset: Int, length: Int = 1)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = if (dimension < 0) input.dim() + dimension + 1 else dimension
    val length = if (this.length < 0) input.size(dim) - offset + this.length + 2 else this.length
    val outputNarrow = input.narrow(dim, offset, length)
    output.resizeAs(outputNarrow).copy(outputNarrow)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dim = if (dimension < 0) input.dim() + dimension + 1 else dimension
    val length = if (this.length < 0) input.size(dim) - offset + this.length + 2 else this.length
    gradInput.resizeAs(input).zero()
    gradInput.narrow(dim, offset, length).copy(gradOutput)
    gradInput
  }
  override def toString(): String = {
    s"${getPrintName}($dimension, $offset, $length)"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Narrow[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Narrow[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode())
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Narrow {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimension: Int,
    offset: Int,
    length: Int = 1)(implicit ev: TensorNumeric[T]) : Narrow[T] = {
    new Narrow[T](dimension, offset, length)
  }
}
