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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Dropout => NNDropout}
import com.intel.analytics.bigdl.tensor.DnnTensor

class Dropout(
  val initP: Double = 0.5,
  val inplace: Boolean = false,
  var scale: Boolean = true) extends MklDnnLayer {
  private val dropout = NNDropout[Float](initP, inplace, scale)
  private var mask: DnnTensor[Float] = _

  private def format(shape: Array[Int], layout: Int): Int = {
    shape.length match {
      case 2 => Memory.Format.nc
      // reminder: for 3 dimension, we should keep original layout (ntc or tnc)
      case 3 => layout
      case 4 => Memory.Format.nchw
      case _ => throw new UnsupportedOperationException(s"${getName()} unsupported input shape")
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = inputs.map(x => HeapData(x.shape, format(x.shape, x.layout)))
    _outputFormats = inputs.map(x => HeapData(x.shape, format(x.shape, x.layout)))
    // we should genereate the primitives here, otherwise the initTensor can't get the padding shape
    _outputFormats.map(_.getPrimitive(runtime))
    output = initTensor(_outputFormats.head)
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = grad.map(x => HeapData(x.shape, format(x.shape, x.layout)))
    _gradOutputFormatsForWeight = grad.map(x => HeapData(x.shape, format(x.shape, x.layout)))
    _gradInputFormats = grad.map(x => HeapData(x.shape, format(x.shape, x.layout)))
    _gradInputFormats.map(_.getPrimitive(runtime))
    gradInput = initTensor(_gradInputFormats.head)
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (isTraining()) {
      output = dropout.updateOutput(input)
    } else {
      output.toTensor[Float].copy(input.toTensor[Float])
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = dropout.updateGradInput(input, gradOutput)
    gradInput
  }

  override def clearState(): this.type = {
    dropout.clearState()
    this
  }

  override def toString(): String = {
    s"mkldnn.Dropout"
  }
}

object Dropout {
  def apply(
    initP: Double = 0.5,
    inplace: Boolean = false,
    scale: Boolean = true) : Dropout = {
    new Dropout(initP, inplace, scale)
  }
}

