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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper

class ReorderMemorySpec extends BigDLSpecHelper {
  "From heap to native" should "be correct" in {
    val layer = ReorderMemory(new NativeData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc))
    layer.setRuntime(new MklDnnRuntime())
    layer.initFwdPrimitives(Array(HeapData(Array(3, 4), Memory.Format.nc)), Phase.TrainingPhase)
    layer.initBwdPrimitives(Array(NativeData(Array(3, 4), Memory.Format.nc)), Phase.TrainingPhase)
    val input = Tensor[Float](3, 4).rand()
    val output = layer.forward(input)
    val grad = layer.backward(input, output)
    grad should be(input)
  }

  "From heap to heap" should "be correct" in {
    val layer = ReorderMemory(
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc),
      HeapData(Array(3, 4), Memory.Format.nc)
    )
    layer.setRuntime(new MklDnnRuntime())
    layer.initFwdPrimitives(Array(HeapData(Array(3, 4), Memory.Format.nc)), Phase.TrainingPhase)
    layer.initBwdPrimitives(Array(NativeData(Array(3, 4), Memory.Format.nc)), Phase.TrainingPhase)
    val input = Tensor[Float](3, 4).rand()
    val output = layer.forward(input)
    val grad = layer.backward(input, output)
    grad should be(input)
  }

  "Reorder from nhwc to nchw" should "be correct" in {
    val shapeNCHW = Array(4, 3, 7, 7)
    val shapeNHWC = Array(4, 7, 7, 3)
    val inputFormats = HeapData(shapeNHWC, Memory.Format.nhwc)
    val outputFormats = HeapData(shapeNCHW, Memory.Format.nchw)
    val gradInputFormats = HeapData(shapeNCHW, Memory.Format.nchw)
    val gradOutputFormats = HeapData(shapeNHWC, Memory.Format.nhwc)

    val layer = ReorderMemory(inputFormat = inputFormats, outputFormat = outputFormats,
      gradInputFormat = gradInputFormats, gradOutputFomat = gradOutputFormats)

    layer.setRuntime(new MklDnnRuntime())
    layer.initFwdPrimitives(Array(inputFormats), Phase.TrainingPhase)
    layer.initBwdPrimitives(Array(gradOutputFormats), Phase.TrainingPhase)


    val input = Tensor[Float](4, 7, 7, 3).rand()
    val gradOutput = input.clone()
    val output = layer.forward(input).toTensor[Float]
    val grad = layer.backward(input, gradOutput)

    val inputNHWC = input.transpose(2, 4).transpose(3, 4).contiguous().clone()

    inputNHWC should be(output)
    inputNHWC should be(grad)
  }

  "Reorder from nchw to nhwc" should "be correct" in {
    val shapeNCHW = Array(4, 3, 7, 7)
    val shapeNHWC = Array(4, 7, 7, 3)
    val inputFormats = HeapData(shapeNCHW, Memory.Format.nchw)
    val outputFormats = HeapData(shapeNHWC, Memory.Format.nhwc)
    val gradInputFormats = HeapData(shapeNHWC, Memory.Format.nhwc)
    val gradOutputFormats = HeapData(shapeNCHW, Memory.Format.nchw)

    val layer = ReorderMemory(inputFormat = inputFormats, outputFormat = outputFormats,
      gradInputFormat = gradInputFormats, gradOutputFomat = gradOutputFormats)

    layer.setRuntime(new MklDnnRuntime())
    layer.initFwdPrimitives(Array(inputFormats), Phase.TrainingPhase)
    layer.initBwdPrimitives(Array(gradOutputFormats), Phase.TrainingPhase)

    val input = Tensor[Float](4, 3, 7, 7).rand()
    val gradOutput = input.clone()
    val output = layer.forward(input).toTensor[Float]
    val grad = layer.backward(input, gradOutput).toTensor[Float]

    val inputNHWC = input.transpose(2, 3).transpose(3, 4).contiguous().clone()

    // grad.resizeAs(inputNHWC)

    inputNHWC should be(output)
    inputNHWC should be(grad)
  }

}
