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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Highway[T: ClassTag](size: Int, activation: String = "tanh")
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var input = Input()
  var gate = Linear(size, size).inputs(input)
  gate = Sigmoid().inputs(gate)
  val negatedGate = AddConstant(-1, true).inputs(Negative(true).inputs(gate))
  var transformed = Linear(size, size).inputs(input)
  transformed = act.inputs(input)
  val transformedGated = CMulTable().inputs(gate, transformed)
  val identityGate = CMulTable().inputs(negatedGate, input)
  val value = CAddTable().inputs(transformedGated, identityGate)
  val highway = Graph(Array(input), Array(value))

  private def act = activation match {
    case "tanh" => Tanh()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.size(2)
    require(dim == size, "the given size is not equal to the input size")
    output = highway.forward(input).toTensor[T]
    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = highway.updateGradInput(input, gradOutput).toTensor[T]
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    highway.accGradParameters(input, gradOutput)
  }


  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    highway.parameters()
  }


  override def getParametersTable(): Table = {
    highway.getParametersTable()
  }
}

object Highway {
  def apply[@specialized(Float, Double) T: ClassTag](size: Int, activation: String = "tanh")
    (implicit ev: TensorNumeric[T]): Highway[T] = new Highway(size, activation)
}
