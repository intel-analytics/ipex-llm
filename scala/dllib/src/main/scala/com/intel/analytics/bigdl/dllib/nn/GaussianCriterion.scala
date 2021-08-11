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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Computes the log-likelihood of a sample x given a Gaussian distribution p.
 */
class GaussianCriterion[@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends AbstractCriterion[Table, Tensor[T], T]  {

  @transient
  private var mean: Tensor[T] = null
  @transient
  private var vari: Tensor[T] = null
  @transient
  private var expVar: Tensor[T] = null

  override def updateOutput(input: Table, target: Tensor[T]): T = {
    if (mean == null) mean = Tensor[T]()
    if (vari == null) vari = Tensor[T]()
    if (expVar == null) expVar = Tensor[T]()
    /*
    log(sigma) + 0.5 *log(2pi) + 0.5 * (x - mu)^2/sigma^2
    input[1] = mu
    input[2] = log(sigma^2)
    */
    mean.resizeAs(input[Tensor[T]](1)).copy(input(1))
    vari.resizeAs(input[Tensor[T]](2)).copy(input(2))
    expVar.resizeAs(input[Tensor[T]](2)).copy(input(2))

    expVar.exp()
    vari.mul(ev.fromType(0.5)).add(ev.fromType(0.5 * math.log(2 * math.Pi)))

    vari.add(ev.fromType(0.5), mean.add(ev.fromType(-1), target).pow(ev.fromType(2)).cdiv(expVar))

    output = vari.sum()
    return output
  }

  override def updateGradInput(input: Table, target: Tensor[T]): Table = {
    if (!gradInput.contains(1)) gradInput(1) = Tensor()
    if (!gradInput.contains(2)) gradInput(2) = Tensor()

    mean.resizeAs(input[Tensor[T]](1)).copy(input(1))
    expVar.resizeAs(input[Tensor[T]](2)).copy(input(2))
    expVar.exp()

    // -(x-mu)/sigma^2
    gradInput[Tensor[T]](1).resizeAs(mean).copy(mean.add(ev.fromType(-1), target))
    gradInput[Tensor[T]](1).cdiv(expVar)
    // 0.5 - 0.5 * (x - mu)^2 / sigma^2
    gradInput(2) = mean.cmul(gradInput[Tensor[T]](1)).mul(ev.fromType(-0.5)).add(ev.fromType(0.5))

    gradInput
  }
}

object GaussianCriterion {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : GaussianCriterion[T] = {
    new GaussianCriterion[T]()
  }
}
