/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

class Recurrent[T : ClassTag] (
  hiddenSize: Int = 3,
  bpttTruncate: Int = 2)
  (implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  val hidden = Tensor[T]()
  var module: Module[T] = _
  var transform: Module[T] = _
  var (batchSize, times) = (0, 0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3,
      "Recurrent: input should be a 3D Tensor, e.g [batch, times, nDim], " +
        s"current input.dim = ${input.dim}")
    require(modules.length == 2,
      "Recurrent: rnn container must include a cell and a non-linear layer, " +
        s"current container length is ${modules.length}")

    module = modules(0)
    transform = modules(1)

    batchSize = input.size(1)
    times = input.size(2)

    output.resize(Array(batchSize, times, hiddenSize))
    hidden.resize(Array(batchSize, times + 1, hiddenSize))

    var i = 1
    while (i <= times) {
      val curInput = T(input.select(2, i), hidden.select(2, i))
      val currentOutput = module.updateOutput(curInput)
      transform.updateOutput(currentOutput)
      output.select(2, i).copy(transform.output.toTensor)
      hidden.select(2, i + 1).copy(transform.output.toTensor)
      i += 1
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    var i = times
    while (i >= 1) {
      transform.output = hidden.select(2, i + 1)
      var deltaHidden = transform.updateGradInput(hidden.select(2, i), gradOutput.select(2, i))
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(input.select(2, bpttStep), hidden.select(2, bpttStep))
        module.accGradParameters(curInput, deltaHidden)
        transform.output.toTensor
          .copy(hidden.select(2, bpttStep))
        deltaHidden = transform.updateGradInput(Tensor(),
          module.updateGradInput(curInput, deltaHidden).toTable(2))
        bpttStep -= 1
      }
      i -= 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    var i = times
    while (i >= 1) {
      transform.output.toTensor
        .copy(hidden.select(2, i + 1))
      var deltaHidden = transform.updateGradInput(hidden.select(2, i), gradOutput.select(2, i))
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(input.select(2, bpttStep), hidden.select(2, bpttStep))
        val gradInputBundle = module.updateGradInput(curInput, deltaHidden).toTable
        gradInput.select(2, bpttStep).add(gradInputBundle(1).asInstanceOf[Tensor[T]])
        transform.output.toTensor
          .copy(hidden.select(2, bpttStep))
        deltaHidden = transform.updateGradInput(Tensor(), gradInputBundle(2))
        bpttStep -= 1
      }
      i -= 1
    }
    gradInput
  }

  override def toString(): String = {
    val str = "nn.Recurrent"
    str
  }
}

object Recurrent {
  def apply[@specialized(Float, Double) T: ClassTag](
    hiddenSize: Int = 3,
    bpttTruncate: Int = 2)
    (implicit ev: TensorNumeric[T]) : Recurrent[T] = {
    new Recurrent[T](hiddenSize, bpttTruncate)
  }
}
