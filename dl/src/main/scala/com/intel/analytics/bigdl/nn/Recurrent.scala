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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class Recurrent[T : ClassTag] (
  hiddenSize: Int = 3,
  bpttTruncate: Int = 2)
  (implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  val hiddens = T()
  var module: Module[T] = _
  var transform: Module[T] = _
  var transforms = ParallelTable()
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
    if (hiddens.length() == 0) {
      transforms.add(transform)
      module.output match {
        case tensor: Tensor[T] => hiddens.insert(Tensor())
        case table: Table =>
          var j = 1
          while (j <= table.length()) {
            hiddens.insert(
              Tensor().resize(Array(batchSize, times + 1, hiddenSize)))
            if (j != 1) {
              transforms.add(transform.cloneModule())
            }
            j += 1
          }
        case _ =>
      }
    }

    var j = 1
    while (j <= hiddens.length()) {
      hiddens[Tensor[T]](j)
        .resize(Array(batchSize, times + 1, hiddenSize))
      j += 1
    }

    var i = 1
    while (i <= times) {
      val curInput = T(input.select(2, i))
      j = 1
      while (j <= hiddens.length()) {
        val hidden = hiddens[Tensor[T]](j)
        curInput.insert(hidden.select(2, i))
        j += 1
      }
      val currentOutput = module.updateOutput(curInput)
      currentOutput match {
        case tensor: Tensor[T] => transforms.updateOutput(T(tensor))
        case table: Table => transforms.updateOutput(table)
        case _ =>
      }
      output.select(2, i).copy(transforms.output.toTable[Tensor[T]](1))
      j = 1
      while (j <= hiddens.length()) {
        val hidden = hiddens[Tensor[T]](j)
        val output = transforms.output.toTable[Tensor[T]](j)
        hidden.select(2, i + 1).copy(output)
        j += 1
      }
      i += 1
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    var i = times
    while (i >= 1) {
      var deltaHidden = T()
      var j = 1
      while (j <= hiddens.length()) {
        val hidden = hiddens[Tensor[T]](j)
        transforms.output.update(j, hidden.select(2, i + 1))
        deltaHidden.insert(
          transform.updateGradInput(hidden.select(2, i), gradOutput.select(2, i))
        )
        j += 1
      }
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        j = 1
        val curInput = T(input.select(2, bpttStep))
        while (j <= hiddens.length()) {
          val hidden = hiddens[Tensor[T]](j)
          curInput.insert(hidden.select(2, bpttStep))
          j += 1
        }
        module.accGradParameters(curInput, getIfTableSizeOne(deltaHidden))

        val moduleGrads = module
          .updateGradInput(curInput, getIfTableSizeOne(deltaHidden)).toTable
        j = 1
        while (j <= hiddens.length()) {
          val hidden = hiddens[Tensor[T]](j)
          val transform = transforms.modules.toArray.apply(j - 1)
          transform.output
            .toTensor
            .copy(hidden.select(2, bpttStep))
          deltaHidden
            .update(j, transform.updateGradInput(Tensor(), moduleGrads[Tensor[T]](j + 1)))
          j += 1
        }
        bpttStep -= 1
      }
      i -= 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    var i = times
    while (i >= 1) {
      val deltaHidden = T()
      var j = 1
      while (j <= hiddens.length()) {
        val hidden = hiddens[Tensor[T]](j)
        transforms.output.update(j, hidden.select(2, i + 1))
        deltaHidden.insert(
          transform.updateGradInput(hidden.select(2, i), gradOutput.select(2, i))
        )
        j += 1
      }
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(input.select(2, bpttStep))
        var j = 1
        while (j <= hiddens.length()) {
          val hidden = hiddens[Tensor[T]](j)
          curInput.insert(hidden.select(2, bpttStep))
          j += 1
        }
        val gradInputBundle = module
          .updateGradInput(curInput,
            if (deltaHidden.length() == 1) deltaHidden(1) else deltaHidden)
          .toTable
        gradInput.select(2, bpttStep).add(gradInputBundle(1).asInstanceOf[Tensor[T]])
        j = 1
        while (j <= hiddens.length()) {
          val hidden = hiddens[Tensor[T]](j)
          val transform = transforms.modules.toArray.apply(j - 1)
          deltaHidden.update(j, transform.updateGradInput(Tensor(), gradInputBundle(j + 1)))
          j += 1
        }
        bpttStep -= 1
      }
      i -= 1
    }
    gradInput
  }

  def getIfTableSizeOne(table: Table): Activity = {
    if (table.length() == 1) table[Tensor[T]](1) else table
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
