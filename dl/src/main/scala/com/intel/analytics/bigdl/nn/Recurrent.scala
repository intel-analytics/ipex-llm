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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

class Recurrent[T : ClassTag] (
  hiddenSize: Int = 3,
  bpttTruncate: Int = 2)
(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  val hidden = T(Tensor[T](hiddenSize))

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 2, "input should be a two dimension Tensor")
    require(modules.length == 2, "rnn container must include a cell and a non-linear layer")

    val module = modules(0)
    val transform = modules(1)

    val numOfWords = input.size(1)
    output.resize(Array(numOfWords, hiddenSize))

    var i = 1
    while (i <= numOfWords) {
      val curInput = T(input(i), hidden(i).asInstanceOf[Tensor[T]])
      val currentOutput = module.updateOutput(curInput)
      transform.updateOutput(currentOutput)
      output.update(i, transform.output.asInstanceOf[Tensor[T]])
      hidden(i + 1) = Tensor[T]()
        .resizeAs(transform.output.asInstanceOf[Tensor[T]])
        .copy(transform.output.asInstanceOf[Tensor[T]])
      i += 1
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    val module = modules(0)
    val transform = modules(1)

    val numOfWords = input.size(1)
    var i = numOfWords
    while (i >= 1) {
      transform.output = hidden(i + 1).asInstanceOf[Tensor[T]]
      var deltaHidden = transform.updateGradInput(hidden(i), gradOutput(i))
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(input(bpttStep), hidden(bpttStep).asInstanceOf[Tensor[T]])
        module.accGradParameters(curInput, deltaHidden)
        transform.output.asInstanceOf[Tensor[T]]
          .copy(hidden(bpttStep).asInstanceOf[Tensor[T]])
        deltaHidden = transform.updateGradInput(Tensor(),
          module.updateGradInput(curInput, deltaHidden).toTable(2))
        bpttStep -= 1
      }
      i -= 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val module = modules(0)
    val transform = modules(1)

    gradInput.resize(input.size).zero

    val numOfWords = input.size(1)
    var i = numOfWords
    while (i >= 1) {
      transform.output.asInstanceOf[Tensor[T]]
        .copy(hidden(i + 1).asInstanceOf[Tensor[T]])
      var deltaHidden = transform.updateGradInput(hidden(i), gradOutput(i))
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(input(bpttStep), hidden(bpttStep).asInstanceOf[Tensor[T]])
        val gradInputBundle = module.updateGradInput(curInput, deltaHidden).toTable
        gradInput(bpttStep).add(gradInputBundle(1).asInstanceOf[Tensor[T]])
        transform.output.asInstanceOf[Tensor[T]]
          .copy(hidden(bpttStep).asInstanceOf[Tensor[T]])
        deltaHidden = transform.updateGradInput(Tensor(), gradInputBundle(2))
        bpttStep -= 1
      }
      i -= 1
    }

    gradInput
  }

  override def toString(): String = {
    var str = "nn.Recurrent"
    str
  }
}

object Recurrent {
  def apply[@specialized(Float, Double) T: ClassTag](
    hiddenSize: Int = 3,
    bpttTruncate: Int = 2)(implicit ev: TensorNumeric[T]) : Recurrent[T] = {
    new Recurrent[T](hiddenSize, bpttTruncate)
  }
}
