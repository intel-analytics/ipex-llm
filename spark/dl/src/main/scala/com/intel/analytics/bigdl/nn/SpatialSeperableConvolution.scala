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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Separable convolutions consist in first performing a depthwise spatial convolution (which acts
 * on each input channel separately) followed by a pointwise convolution which mixes together the
 * resulting output channels. The  depthMultiplier argument controls how many output channels are
 * generated per input channel in the depthwise step.
 *
 * @param nInputChannel
 * @param nOutputChannel
 * @param depthMultiplier
 * @param kW
 * @param kH
 * @param sW
 * @param sH
 * @param pW
 * @param pH
 * @param hasBias
 * @param dataFormat
 * @param wRegularizer
 * @param bRegularizer
 * @param pRegularizer
 * @tparam T Numeric type. Only support float/double now
 */
class SpatialSeperableConvolution[T: ClassTag](
  val nInputChannel: Int,
  val nOutputChannel: Int,
  val depthMultiplier: Int,
  val kW: Int, val kH: Int,
  val sW: Int, val sH: Int,
  val pW: Int, val pH: Int,
  val hasBias: Boolean,
  val dataFormat: DataFormat,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  var pRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[T], T]{

  private val internalChannel = nInputChannel * depthMultiplier

  private val channelDim = if (dataFormat == DataFormat.NCHW) 2 else 4

  private val depthWeight = if (dataFormat == DataFormat.NCHW) {
    Tensor[T](depthMultiplier, nInputChannel, kW, kH)
  } else {
    Tensor[T](kW, kH, nInputChannel, depthMultiplier)
  }

  private val depthGradWeight = Tensor[T].resizeAs(depthWeight)

  private val pointWeight = if (dataFormat == DataFormat.NCHW) {
    Tensor[T](nOutputChannel, internalChannel, 1, 1)
  } else {
    Tensor[T](1, 1, internalChannel, nOutputChannel)
  }

  private val pointGradWeight = Tensor[T].resizeAs(pointWeight)

  private val bias = Tensor[T](nOutputChannel)

  private val gradBias = Tensor[T].resizeAs(bias)

  private val depthConv = SpatialConvolution[T](
    nInputPlane = nInputChannel,
    nOutputPlane = internalChannel,
    kernelW = kW,
    kernelH = kH,
    strideW = sW,
    strideH = sH,
    padW = pW,
    padH = pH,
    wRegularizer = wRegularizer,
    bRegularizer = null,
    withBias = false,
    format = dataFormat
  )

  private val pointWiseConv2D = SpatialConvolution[T](
    nInputPlane = internalChannel,
    nOutputPlane = nOutputChannel,
    kernelW = 1,
    kernelH = 1,
    strideW = 1,
    strideH = 1,
    padW = 0,
    padH = 0,
    wRegularizer = pRegularizer,
    bRegularizer = bRegularizer,
    withBias = hasBias,
    format = dataFormat,
    initWeight = pointWeight,
    initGradWeight = pointGradWeight,
    initBias = bias,
    initGradBias = gradBias
  )

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(depthWeight, pointWeight, bias), Array(depthGradWeight, pointGradWeight, gradBias))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "SpatialSeperableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeperableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    // Copy weight
    depthConv.weight.zero()
    var in = 0
    while(in < input.size(channelDim)) {
      var out = 0
      while(out < depthMultiplier) {
        depthConv.weight.select(4, in + 1).select(4, in * depthMultiplier + out + 1)
          .copy(depthWeight.select(3, in + 1).select(3, out + 1))
        out += 1
      }
      in += 1
    }

    depthConv.forward(input)
    output = pointWiseConv2D.forward(depthConv.output)
    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "SpatialSeperableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeperableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    require(gradOutput.nDimension() == 4, "SpatialSeperableConvolution only accept 4D gradOutput")
    require(gradOutput.isContiguous(), "SpatialSeperableConvolution require contiguous gradOutput")
    require(nOutputChannel == gradOutput.size(channelDim),
      "gradOutput tensor channel dimension size doesn't match layer nOutputChannel")

    pointWiseConv2D.backward(depthConv.output, gradOutput)
    gradInput = depthConv.backward(input, pointWiseConv2D.gradInput)
    copyDepthGradWeight()
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "SpatialSeperableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeperableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    require(gradOutput.nDimension() == 4, "SpatialSeperableConvolution only accept 4D gradOutput")
    require(gradOutput.isContiguous(), "SpatialSeperableConvolution require contiguous gradOutput")
    require(nOutputChannel == gradOutput.size(channelDim),
      "gradOutput tensor channel dimension size doesn't match layer nOutputChannel")

    pointWiseConv2D.updateGradInput(depthConv.output, gradOutput)
    gradInput = depthConv.updateGradInput(input, pointWiseConv2D.gradInput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.nDimension() == 4, "SpatialSeperableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeperableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    require(gradOutput.nDimension() == 4, "SpatialSeperableConvolution only accept 4D gradOutput")
    require(gradOutput.isContiguous(), "SpatialSeperableConvolution require contiguous gradOutput")
    require(nOutputChannel == gradOutput.size(channelDim),
      "gradOutput tensor channel dimension size doesn't match layer nOutputChannel")

    pointWiseConv2D.accGradParameters(depthConv.output, gradOutput)
    depthConv.accGradParameters(input, pointWiseConv2D.gradInput)
    copyDepthGradWeight()
  }

  private def copyDepthGradWeight(): Unit = {
    var in = 0
    while(in < nInputChannel) {
      var out = 0
      while(out < depthMultiplier) {
        depthGradWeight.select(3, in + 1).select(3, out + 1)
          .copy(depthConv.gradWeight.select(4, in + 1).select(4, in * depthMultiplier + out + 1))
        out += 1
      }
      in += 1
    }
  }
}

object SpatialSeperableConvolution {
  def apply[T: ClassTag](nInputChannel: Int, nOutputChannel: Int, depthMultiplier: Int,
    kW: Int, kH: Int, sW: Int = 1, sH: Int = 1, pW: Int = 0, pH: Int = 0,
    hasBias: Boolean = true, dataFormat: DataFormat = DataFormat.NCHW,
    wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
    pRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T])
  : SpatialSeperableConvolution[T] = new SpatialSeperableConvolution(nInputChannel,
    nOutputChannel, depthMultiplier, kW, kH, sW, sH, pW, pH, hasBias, dataFormat, wRegularizer,
    bRegularizer)
}
