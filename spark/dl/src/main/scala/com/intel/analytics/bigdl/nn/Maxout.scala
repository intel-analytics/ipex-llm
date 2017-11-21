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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * [[Maxout]] Use a mask value to skip timesteps for a sequence
 *
 * @param inputSize mask value
 */
class Maxout[T: ClassTag](inputSize: Int, outputSize: Int, maxoutNumber: Int)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val lineart = Linear(inputSize, outputSize * maxoutNumber)
  val viewt = View(maxoutNumber, outputSize).setNumInputDims(1)
  val maxt = Max(1, 2)
  val layer = Sequential().add(lineart)
    .add(viewt)
    .add(maxt)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
//    output = layer.updateOutput(input)
    val o1 = lineart.forward(input)
    val o2 = viewt.forward(o1)
    output = maxt.forward(o2)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = layer.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    layer.accGradParameters(input, gradOutput)
  }

  override def zeroGradParameters(): Unit = {
    layer.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    layer.parameters()
  }

  override def getParametersTable(): Table = {
    layer.getParametersTable()
  }
}

object Maxout {
  def apply[T : ClassTag](inputSize: Int, outputSize: Int, maxoutNumber: Int)
    (implicit ev: TensorNumeric[T]): Maxout[T]
    = new Maxout[T](inputSize, outputSize, maxoutNumber)
}
