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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class KLDCriterion[@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends AbstractCriterion[Table, Tensor[T], T] {

  @transient
  private val mean = Tensor[T]()
  @transient
  private val vari = Tensor[T]()
  @transient
  private val expVar = Tensor[T]()

  override def updateOutput(input: Table, target: Tensor[T]): T = {
    mean.resizeAs(input[Tensor[T]](1)).copy(input(1))
    vari.resizeAs(input[Tensor[T]](2)).copy(input(2))

    //  Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    mean.pow(ev.fromType(2))
    expVar.resizeAs(vari).copy(vari)
    expVar.exp().add(ev.fromType(1)).add(ev.fromType(-1), mean).add(ev.fromType(-1), vari)

    output = ev.times(ev.fromType(0.5), expVar.sum())
    output
  }

  override def updateGradInput(input: Table, target: Tensor[T]): Table = {
    if (!gradInput.contains(1)) gradInput(1) = Tensor()
    if (!gradInput.contains(2)) gradInput(2) = Tensor()

    mean.resizeAs(input[Tensor[T]](1)).copy(input(1))
    vari.resizeAs(input[Tensor[T]](2)).copy(input(2))

    // d_L/d_mu = mu
    gradInput[Tensor[T]](1).resizeAs(mean).copy(mean)
    // d_L/d_sigma = 0.5*(exp(log_sq_sigma)-1)
    gradInput(2) = vari.exp().add(ev.fromType(-1)).mul(ev.fromType(0.5))

    gradInput
  }
}

object KLDCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    implicit ev: TensorNumeric[T]): KLDCriterion[T] = {
    new KLDCriterion[T]()
  }
}