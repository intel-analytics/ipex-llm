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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Takes {mean, log_variance} as input and samples from the Gaussian distribution
 */
class GaussianSampler[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  val eps = Tensor[T]()

  override def updateOutput(input: Table): Tensor[T] = {
    eps.resizeAs(input(1)).randn()
    val output2 = output.toTensor
    output2.resizeAs(input(2)).copy(input(2))
    output2.mul(ev.fromType(0.5)).exp().cmul(eps)
    output2.add(input[Tensor[T]](1))
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    if (!gradInput.contains(1)) gradInput(1) = Tensor()
    if (!gradInput.contains(2)) gradInput(2) = Tensor()

    gradInput[Tensor[T]](1).resizeAs(gradOutput).copy(gradOutput)
    gradInput[Tensor[T]](2).resizeAs(gradOutput).copy(input(2))

    gradInput[Tensor[T]](2).mul(ev.fromType(0.5)).exp().mul(ev.fromType(0.5)).cmul(eps)
    gradInput[Tensor[T]](2).cmul(gradOutput)

    gradInput
  }

  override def clearState() : this.type = {
    super.clearState()
    eps.set()
    this
  }
}

object GaussianSampler {
  def apply[@specialized(Float, Double) T: ClassTag]()(
   implicit ev: TensorNumeric[T]) : GaussianSampler[T] = {
    new GaussianSampler[T]()
  }
}
