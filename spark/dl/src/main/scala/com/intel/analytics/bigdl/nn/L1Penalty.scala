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
 * adds an L1 penalty to an input (for sparsity).
 * L1Penalty is an inline module that in its forward propagation copies the input Tensor
 * directly to the output, and computes an L1 loss of the latent state (input) and stores
 * it in the module's loss field. During backward propagation: gradInput = gradOutput + gradLoss.
 *
 * @param l1weight
 * @param sizeAverage
 * @param provideOutput
 */

@SerialVersionUID(- 6261350003722613506L)
class L1Penalty[T: ClassTag]
 (val l1weight: Int, val sizeAverage: Boolean = false,
 val provideOutput: Boolean = true)
 (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var loss: T = ev.fromType(0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var m: Double = l1weight
    if (sizeAverage) m = m / input.nElement()
    loss = ev.times(ev.fromType(m), input.norm(1))
    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    var m: Double = l1weight
    if (sizeAverage) m = m / input.nElement()
    gradInput.resizeAs(input).copy(input).sign().mul(ev.fromType(m))

    if (provideOutput) gradInput.add(gradOutput)
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[L1Penalty[T]]

  override def equals(other: Any): Boolean = other match {
    case that: L1Penalty[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        l1weight == that.l1weight &&
        sizeAverage == that.sizeAverage &&
        provideOutput == that.provideOutput
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), l1weight, sizeAverage, provideOutput)
    state.map(getHashCode).foldLeft(0)((a, b) => 37 * a + b)
  }

  override def toString(): String = {
    s"${getPrintName}($l1weight, $sizeAverage, $provideOutput)"
  }
}

object L1Penalty {
  def apply[@specialized(Float, Double) T: ClassTag](
      l1weight: Int,
      sizeAverage: Boolean = false,
      provideOutput: Boolean = true)(implicit ev: TensorNumeric[T]) : L1Penalty[T] = {
    new L1Penalty[T](l1weight, sizeAverage, provideOutput)
  }
}
