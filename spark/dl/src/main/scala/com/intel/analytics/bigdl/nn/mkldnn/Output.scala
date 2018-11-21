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
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

/**
 * Convert output to user defined layout
 * @param outputLayOut output memory layout
 * @param gradOutputLayout gradoutput memory layout
 */
class Output(outputLayOut: Int = Memory.Format.nc,
             gradOutputLayout: Int = Memory.Format.nc) extends MklDnnLayer {

  override private[bigdl] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length == 1, "Only accept one tensor as input")
    require(inputs(0).shape.length == 4 || inputs(0).shape.length == 2)

    val inputShape = inputs(0).shape
    val inputLayout = inputs(0).layout

    if (inputLayout != outputLayOut) {
      val outputShape = if (outputLayOut == Memory.Format.nhwc) {
        // nchw -> nhwc
        Array(inputShape(0), inputShape(2), inputShape(3), inputShape(1))
      } else if (inputLayout == Memory.Format.nhwc) {
        // nhwc -> nchw
        Array(inputShape(0), inputShape(3), inputShape(1), inputShape(2))
      } else inputShape
      _outputFormats = Array(HeapData(outputShape, outputLayOut))
      _inputFormats = _outputFormats
    } else {
      _outputFormats = inputs
      _inputFormats = inputs
    }

    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (input.toTensor[Float].isInstanceOf[DnnTensor[Float]]) {
      if (output == null) output = Tensor[Float]()
      output.toTensor[Float].resize(input.toTensor[Float].size()).copy(input.toTensor[Float])
    } else {
      output = input
    }
    output
  }

  override private[bigdl] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == 1, "Only accept one tensor as input")
    require(grads(0).shape.length == 4 || grads(0).shape.length == 2)

    val inputShape = grads(0).shape
    val inputLayout = grads(0).layout

    if (inputLayout != gradOutputLayout) {
      val outputShape = if (gradOutputLayout == Memory.Format.nhwc) {
        // nchw -> nhwc
        Array(inputShape(0), inputShape(2), inputShape(3), inputShape(1))
      } else if (inputLayout == Memory.Format.nhwc) {
        // nhwc -> nchw
        Array(inputShape(0), inputShape(3), inputShape(1), inputShape(2))
      } else inputShape

      _gradInputFormats = Array(HeapData(outputShape, gradOutputLayout))
      _gradOutputFormats = _gradInputFormats
    } else {
      _gradInputFormats = grads
      _gradOutputFormats = grads
    }
    _gradOutputFormatsForWeight = _gradOutputFormats
    (_gradInputFormats, _gradOutputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (gradOutput.toTensor[Float].isInstanceOf[DnnTensor[Float]]) {
      if (gradInput == null) gradInput = Tensor[Float]()
      gradInput.toTensor[Float].resize(gradOutput.toTensor[Float].size())
        .copy(gradOutput.toTensor[Float])
    } else {
      gradInput = gradOutput
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.mkl.Output(${outputLayOut}, ${gradOutputLayout})"
  }
}

object Output {
  def apply(outputLayOut: Int = Memory.Format.nc,
            gradOutputLayout: Int = Memory.Format.nc): Output =
    new Output(outputLayOut, gradOutputLayout)
}
