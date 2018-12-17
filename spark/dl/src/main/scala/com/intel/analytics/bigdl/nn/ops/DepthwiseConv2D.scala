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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.{SpatialConvolution, SpatialSeparableConvolution}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.tf.loaders.Adapter

import scala.reflect.ClassTag

class DepthwiseConv2D[T: ClassTag](
  strideW: Int, strideH: Int,
  padW: Int, padH: Int,
  dataFormat: DataFormat
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private var conv: SpatialConvolution[T] = _
  private var channelMultiplier = 0

  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)
    val channelDim = if (dataFormat == DataFormat.NHWC) 4 else 2
    val kHDim = if (dataFormat == DataFormat.NHWC) 1 else 3
    val kWDim = if (dataFormat == DataFormat.NHWC) 2 else 4

    if (conv == null) {
      channelMultiplier = filter.size(channelDim)
      conv = SpatialConvolution(
        nInputPlane = input.size(channelDim),
        nOutputPlane = channelMultiplier * input.size(channelDim),
        kernelH = filter.size(kHDim),
        kernelW = filter.size(kWDim),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        withBias = false,
        format = dataFormat
      )
      conv.weight.zero()
    }

    SpatialSeparableConvolution.copyWeight(conv.weight, input.size(channelDim), channelMultiplier,
      filter, dataFormat)
    output = conv.forward(input)
    output
  }
}

object DepthwiseConv2D {
  def apply[T: ClassTag](
    strideW: Int, strideH: Int,
    padW: Int, padH: Int,
    dataFormat: DataFormat = DataFormat.NHWC
  )(implicit ev: TensorNumeric[T]): DepthwiseConv2D[T] =
    new DepthwiseConv2D(strideW, strideH, padW, padH, dataFormat)
}

private[bigdl] class DepthwiseConv2DBackpropInput[T: ClassTag](
  strideW: Int, strideH: Int,
  padW: Int, padH: Int,
  dataFormat: DataFormat
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private var conv: SpatialConvolution[T] = _
  private var channelMultiplier = 0
  private val dummyInput = Tensor[T]()

  override def updateOutput(inputs: Table): Tensor[T] = {
    val inputSize: Tensor[Int] = inputs[Tensor[Int]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)
    val gradOutput: Tensor[T] = inputs[Tensor[T]](3)
    val channelDim = if (dataFormat == DataFormat.NHWC) 4 else 2
    val kHDim = if (dataFormat == DataFormat.NHWC) 1 else 3
    val kWDim = if (dataFormat == DataFormat.NHWC) 2 else 4
    dummyInput.resize(inputSize.toArray())

    if (conv == null) {
      channelMultiplier = filter.size(4)
      conv = SpatialConvolution(
        nInputPlane = inputSize.valueAt(channelDim),
        nOutputPlane = channelMultiplier * inputSize.valueAt(channelDim),
        kernelH = filter.size(kHDim),
        kernelW = filter.size(kWDim),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        withBias = false,
        format = dataFormat
      )
      conv.weight.zero()
      conv.forward(dummyInput)
    }

    SpatialSeparableConvolution.copyWeight(conv.weight, inputSize.valueAt(channelDim),
      channelMultiplier, filter, dataFormat)
    output = conv.updateGradInput(dummyInput, gradOutput)
    output
  }
}

private[bigdl] object DepthwiseConv2DBackpropInput {
  def apply[T: ClassTag](
    strideW: Int, strideH: Int,
    padW: Int, padH: Int,
    dataFormat: DataFormat
  )(implicit ev: TensorNumeric[T]): DepthwiseConv2DBackpropInput[T] =
    new DepthwiseConv2DBackpropInput(strideW, strideH, padW, padH, dataFormat)
}

private[bigdl] class DepthwiseConv2DBackpropFilter[T: ClassTag](
  strideW: Int, strideH: Int,
  padW: Int, padH: Int,
  dataFormat: DataFormat
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private var conv: SpatialConvolution[T] = _
  private var channelMultiplier = 0

  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val filterSize: Tensor[Int] = inputs[Tensor[Int]](2)
    val gradOutput: Tensor[T] = inputs[Tensor[T]](3)
    val channelDim = if (dataFormat == DataFormat.NHWC) 4 else 2
    val kHDim = if (dataFormat == DataFormat.NHWC) 1 else 3
    val kWDim = if (dataFormat == DataFormat.NHWC) 2 else 4


    if (conv == null) {
      channelMultiplier = filterSize.valueAt(4)
      conv = SpatialConvolution(
        nInputPlane = input.size(channelDim),
        nOutputPlane = channelMultiplier * input.size(channelDim),
        kernelH = filterSize.valueAt(kHDim),
        kernelW = filterSize.valueAt(kWDim),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        withBias = false,
        format = dataFormat
      )
    }

    conv.forward(input)
    conv.zeroGradParameters()
    conv.accGradParameters(input, gradOutput)
    output.resize(filterSize.toArray())

    SpatialSeparableConvolution.copyDepthGradWeight(input.size(channelDim), channelMultiplier,
      conv.gradWeight, output, dataFormat)

    output
  }
}

private[bigdl] object DepthwiseConv2DBackpropFilter {
  def apply[T: ClassTag](
    strideW: Int, strideH: Int,
    padW: Int, padH: Int,
    dataFormat: DataFormat
  )(implicit ev: TensorNumeric[T]): DepthwiseConv2DBackpropFilter[T] =
    new DepthwiseConv2DBackpropFilter(strideW, strideH, padW, padH, dataFormat)
}
