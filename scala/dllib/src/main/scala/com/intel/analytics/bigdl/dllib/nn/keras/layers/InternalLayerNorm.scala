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

import com.intel.analytics.bigdl.nn.{Mean, Sum}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[zoo] class InternalLayerNorm[T: ClassTag](val nOutput: Int = 768, val eps: Double = 1e-5)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T]{
  val weight = Tensor.ones[T](nOutput).view(1, nOutput)
  val bias = Tensor[T](nOutput).view(1, nOutput)

  var gradWeight: Tensor[T] = Tensor[T]()
  var gradBias: Tensor[T] = Tensor[T]()

  var y: Tensor[T] = null
  var divInput1: Tensor[T] = null
  var divInput2: Tensor[T] = null
  var sqrtInput: Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val u = input.sum(dim).div(ev.fromType(input.size(dim)))
    divInput1 = input.clone().sub(u) // x - u
    val square = divInput1.clone().square()
    val s = square.sum(square.dim()).div(ev.fromType(square.size(square.dim())))
    sqrtInput = s.add(ev.fromType(eps))
    divInput2 = sqrtInput.clone().sqrt()
    y = divInput1.clone.div(divInput2)
    output = y.clone().cmul(weight).add(bias)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val divGradInput1 = gradOutput.clone().cmul(weight).div(divInput2)
//  below code is equal to
//  val divGradInput2 = (divGradInput1.clone().div(divInput2))
// .mul(ev.fromType(-1)).cmul(divInput1)
//  val squareGadO = divGradInput2.sum(divGradInput2.dim())
//  val sqrtGradI = divInput2.div(sqrtInput).mul(ev.fromType(0.5)).cmul(squareGadO)
//  val sumGradI = sqrtGradI.div(ev.fromType(divInput1.size(divInput1.dim())))
//    .expand(divInput1.size())
//  val squareGradI = divInput1.mul(ev.fromType(2)).cmul(sumGradI)
    val divGradInput2 = (divGradInput1.clone().div(divInput2)).cmul(divInput1)
    val squareGadO = divGradInput2.sum(divGradInput2.dim())
    val sqrtGradI = divInput2.div(sqrtInput).cmul(squareGadO)
    val sumGradI = sqrtGradI.div(ev.fromType(-1 * divInput1.size(divInput1.dim())))
     .expand(divInput1.size())
    val squareGradI = divInput1.cmul(sumGradI)

    val addGradO = divGradInput1.add(squareGradI)
    val addGradI = addGradO.sum(addGradO.dim())
    val negativeGradO = addGradI.sum(addGradI.dim())
//    val negativeGradI = negativeGradO.mul(ev.fromType(-1))
    val sum2GradI = negativeGradO.div(ev.fromType(-1 * input.size(input.dim())))

    gradInput = sum2GradI.add(addGradO)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    var i = 1
    gradWeight = y.clone().cmul(gradOutput)
    gradBias = gradOutput
    while (i < gradOutput.dim()) {
      gradBias = gradBias.sum(i)
      gradWeight = gradWeight.sum(i)
      i += 1
    }
    gradBias.resize(bias.size())
    gradWeight.resize(weight.size())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}
