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

private[bigdl] class InputWrapper extends MklDnnLayer {

  private var inputLayer : Input = null

  override private[bigdl] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length == 1, "Only accept one tensor as input")
    inputLayer = Input(inputs(0).shape, inputs(0).layout)
    inputLayer.setRuntime(this.runtime)
    inputLayer.initFwdPrimitives(inputs, phase)
    _inputFormats = inputLayer.inputFormats()
    _outputFormats = inputLayer.outputFormats()
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    output = inputLayer.forward(input)
    output
  }

  override private[bigdl] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == 1, "Only accept one tensor as input")
    inputLayer.initBwdPrimitives(grads, phase)
    _gradInputFormats = inputLayer.gradInputFormats()
    _gradOutputFormats = inputLayer.gradOutputFormats()
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = inputLayer.backward(input, gradOutput)
    gradInput
  }

  override def toString(): String = {
    s"nn.mkl.InputWrapper"
  }

  override def release(): Unit = {
    super.release()
    if(inputLayer != null) {
      inputLayer.release()
    }
  }
}
