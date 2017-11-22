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


import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


import scala.reflect.ClassTag

/**
 * Densely connected highway network.
 * Highway layers are a natural extension of LSTMs to feedforward networks.
 * @param size input size
 * @param withBias whether to include a bias
 * @param activation name of activation function to use
 * @tparam T Numeric type. Only support float/double now
 */
class Highway[T: ClassTag](size: Int, withBias: Boolean = true,
  activation: String = null)
  (implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  var input = Input()
  val l1 = Linear(size, size, withBias = withBias).inputs(input)
  val transformWeight = Sigmoid().inputs(l1)
  val negatedGate = AddConstant(1).inputs(Negative().inputs(transformWeight))
  val l2 = Linear(size, size, withBias = withBias).inputs(input)
  val transformed = if (null != act) act.inputs(l2) else l2
  val transformedGated = CMulTable().inputs(transformWeight, transformed)
  val identityGate = CMulTable().inputs(negatedGate, input)
  val value = CAddTable().inputs(transformedGated, identityGate)
  val highway = Graph(Array(input), Array(value))

  modules.append(highway)

  private def act = activation match {
    case "tanh" => Tanh()
    case _ => null
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.size(2)
    require(dim == size, "the given size is not equal to the input size")
    output = highway.updateOutput(input).toTensor[T]
    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = highway.updateGradInput(input, gradOutput).toTensor[T]
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    highway.accGradParameters(input, gradOutput)
  }
}

object Highway {
  def apply[@specialized(Float, Double) T: ClassTag](size: Int, withBias: Boolean = true,
    activation: String = null)
    (implicit ev: TensorNumeric[T]): Highway[T] = new Highway(size, withBias, activation)
}
