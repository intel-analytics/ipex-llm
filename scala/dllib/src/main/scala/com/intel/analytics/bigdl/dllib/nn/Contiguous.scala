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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * used to make input, gradOutput both contiguous
 */

@SerialVersionUID(- 4704727587714736531L)
class Contiguous[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = input.contiguous()
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]) : Tensor[T] = {
    gradInput = gradOutput.contiguous()
    gradInput
  }

  override def toString(): String = {
    s"nn.Contiguous"
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Contiguous[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Contiguous[T] =>
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

object Contiguous {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Contiguous[T] = {
    new Contiguous[T]()
  }
}
