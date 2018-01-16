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
 *  an element-wise abs operation
 */
@SerialVersionUID(3070101246787506364L)
class Abs[T: ClassTag]
 (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.abs(input)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isContiguous() && gradOutput.isContiguous(),
      "Abs: input and gradOutput should be contiguous")
    gradInput.resizeAs(input).copy(gradOutput)

    val inputArray = input.storage().array()
    val gradArray = gradInput.storage().array()
    val gradOffset = gradInput.storageOffset() - 1

    var i = 0
    while(i < gradInput.nElement()) {
      val g = gradArray(i)
      val z = inputArray(i)
      gradArray(i + gradOffset) = ev.times(g,
        if (ev.isGreater(z, ev.fromType(0))) ev.fromType(1) else ev.fromType(-1))
      i += 1
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Abs[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Abs[T] =>
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

object Abs {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Abs[T] = {
    new Abs[T]()
  }
}
