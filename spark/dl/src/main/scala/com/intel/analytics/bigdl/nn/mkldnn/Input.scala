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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

class Input(shape: Array[Int], layout: Int) extends MklDnnLayer {
  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _outputFormats = Array(HeapData(shape, layout))
    _inputFormats = inputs
    (inputs, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradInputFormats = Array(HeapData(shape, layout))
    _gradOutputFormats = grad
    _gradOutputFormatsForWeight = grad
    (grad, _gradInputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    output = input
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = gradOutput
    gradInput
  }
}

object Input {
  def apply(shape: Array[Int], layout: Int): Input = new Input(shape, layout)
}
