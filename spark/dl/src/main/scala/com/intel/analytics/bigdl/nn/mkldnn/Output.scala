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
 * Convert output to user defined layout and appoint gradOutput layout
 * @param outputLayOut output memory layout
 * @param gradOutputLayout gradoutput memory layout, if it is -1,
 *                         that means gradOutput memory layout is same with output memory layout
 */
class Output(outputLayOut: Int = Memory.Format.nc,
             gradOutputLayout: Int = -1) extends MklDnnLayer {

  private val _outputLayOut = outputLayOut
  private val _gradOutputLayout = if (gradOutputLayout == -1) outputLayOut else gradOutputLayout


  private def getShape(inLayout: Int, inShape: Array[Int], outLayout: Int): Array[Int] = {
    val outputShape =
      if (outLayout == Memory.Format.nhwc && inLayout != Memory.Format.nhwc) {
        // nchw*  -> nhwc
        Array(inShape(0), inShape(2), inShape(3), inShape(1))
      } else if ((outLayout != Memory.Format.nhwc) && (inLayout == Memory.Format.nhwc)) {
        // nhwc -> nchw*
        Array(inShape(0), inShape(3), inShape(1), inShape(2))
      } else if (outLayout == Memory.Format.tnc && inLayout == Memory.Format.ntc) {
        // ntc -> tnc
        Array(inShape(1), inShape(0), inShape(2))
      } else if (outLayout == Memory.Format.ntc && inLayout == Memory.Format.tnc) {
        // tnc -> ntc
        Array(inShape(1), inShape(0), inShape(2))
      } else inShape
    outputShape
  }

  override private[bigdl] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length == 1, "Only accept one tensor as input")
    require(inputs(0).shape.length == 4 || inputs(0).shape.length == 2
      || inputs(0).shape.length == 3,
      s"Only support input with 2 or 3 or 4 dimentions, but get ${inputs(0).shape.length}")

    val outputShape = getShape(inputs(0).layout, inputs(0).shape, _outputLayOut)
    // remind: output memory storage should be heapData
    _outputFormats = Array(HeapData(outputShape, outputLayOut))
    _inputFormats = _outputFormats

    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    output = input
    output
  }

  override private[bigdl] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    require(grads.length == 1, "Only accept one tensor as input")
    require(grads(0).shape.length == 4 || grads(0).shape.length == 2
      || grads(0).shape.length == 3,
      s"Only support gradOutput with 2 or 3 or 4 dimentions, but get ${grads(0).shape.length}")

    val outputShape = getShape(grads(0).layout, grads(0).shape, _gradOutputLayout)

    _gradInputFormats = Array(HeapData(outputShape, _gradOutputLayout))
    _gradOutputFormats = _gradInputFormats
    _gradOutputFormatsForWeight = _gradOutputFormats

    (_gradInputFormats, _gradOutputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = gradOutput
    gradInput
  }

  override def toString(): String = {
    s"nn.mkl.Output(${outputLayOut}, ${gradOutputLayout})"
  }
}

object Output {
  def apply(outputLayOut: Int = Memory.Format.nc,
            gradOutputLayout: Int = -1): Output =
    new Output(outputLayOut, gradOutputLayout)
}
