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

/**
 * @param hiddenSize
 * @param bpttTruncate
 * @param timeDim the time dimension of the input Tensor
 * @param ev
 * @tparam T
 */

class Recurrent[T : ClassTag] (
  hiddenSize: Int = 3,
  bpttTruncate: Int = 2,
  protected val timeDim: Int = 2)
  (implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  /**
   * The Recurrent will always go along the first dimension of the Tensor given that
   * the legacyMode is set to be false in order to boost the performance.
   *
   * If the timeDim is 2, this layer will transpose the first two dimensions of the input.
   * Otherwise, the input will remain unchanged.
   */

  require(timeDim == 1 | timeDim == 2,
    "In Recurrent: the timeDim should be 1 or 2," +
      s"Current timeDim = ${timeDim}")

  protected var legacyMode: Boolean = false
  val hidden = Tensor[T]()
  var module: Module[T] = _
  var transform: Module[T] = _
  var (batchSize, times, nDim) = (0, 0, 0)

  /**
   * batchDim should be either 1 or 2 according to timeDim
   * hiddenSelect is always equal to 1
   * outputSelect is always equal to timeDim
   */

  val batchDim = 3 - timeDim
  val hiddenSelect = 1
  protected var dataSelect = 1
  val outputSelect = timeDim
  var fInput: Tensor[T] = _
  var fGradOutput: Tensor[T] = _

  /**
   * The legacyMode is set only for Unit Test
   */
  private def setLegacyMode(self: Recurrent[T]): Unit = {
    require(self.timeDim == 2,
      "In Recurrent: timeDim should be 2 when legacyMode is set.")
    self.legacyMode = true
    self.dataSelect = 2
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3,
      "Recurrent: input should be a 3D Tensor, e.g [batch, times, nDim], " +
        s"current input.dim = ${input.dim}")
    require(modules.length == 2,
      "Recurrent: rnn container must include a cell and a non-linear layer, " +
        s"current container length is ${modules.length}")

    module = modules(0)
    transform = modules(1)

    batchSize = input.size(batchDim)
    times = input.size(timeDim)

    hidden.resize(Array(times + 1, batchSize, hiddenSize))

    if (dataSelect != timeDim) {
      if (train) {
        fInput = input.transpose(batchDim, timeDim).contiguous
      } else {
        fInput = input.transpose(batchDim, timeDim)
      }
    } else {
      fInput = input
    }

    output.resize(Array(input.size(1), input.size(2), hiddenSize))

    var i = 1
    while (i <= times) {
      val curInput = T(fInput.select(dataSelect, i), hidden.select(hiddenSelect, i))
      val currentOutput = module.updateOutput(curInput)
      transform.updateOutput(currentOutput)
      output.select(outputSelect, i).copy(transform.output.toTensor)
      hidden.select(hiddenSelect, i + 1).copy(transform.output.toTensor)
      i += 1
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    var i = times
    while (i >= 1) {
      transform.output = hidden.select(hiddenSelect, i + 1)
      var deltaHidden = transform.updateGradInput(
        hidden.select(hiddenSelect, i), fGradOutput.select(dataSelect, i))
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(
          fInput.select(dataSelect, bpttStep), hidden.select(hiddenSelect, bpttStep))
        module.accGradParameters(curInput, deltaHidden)
        transform.output.toTensor
          .copy(hidden.select(hiddenSelect, bpttStep))
        deltaHidden = transform.updateGradInput(Tensor(),
          module.updateGradInput(curInput, deltaHidden).toTable(2))
        bpttStep -= 1
      }
      i -= 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    if (dataSelect != timeDim) {
      if (train) {
        fGradOutput = gradOutput.transpose(batchDim, timeDim).contiguous()
      } else {
        fGradOutput = gradOutput.transpose(batchDim, timeDim)
      }
    } else {
      fGradOutput = gradOutput
    }

    var i = times
    while (i >= 1) {
      transform.output.toTensor
        .copy(hidden.select(hiddenSelect, i + 1))
      var deltaHidden = transform.updateGradInput(
        hidden.select(hiddenSelect, i), fGradOutput.select(dataSelect, i))
      var bpttStep = i
      while (bpttStep >= Math.max(1, i - bpttTruncate)) {
        val curInput = T(fInput.select(dataSelect, bpttStep), hidden.select(hiddenSelect, bpttStep))
        val gradInputBundle = module.updateGradInput(curInput, deltaHidden).toTable
        gradInput.select(outputSelect, bpttStep).add(gradInputBundle(1).asInstanceOf[Tensor[T]])
        transform.output.toTensor
          .copy(hidden.select(hiddenSelect, bpttStep))
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
    bpttTruncate: Int = 2,
    timeDim: Int = 2)
    (implicit ev: TensorNumeric[T]) : Recurrent[T] = {
    new Recurrent[T](hiddenSize, bpttTruncate, timeDim)
  }
}
