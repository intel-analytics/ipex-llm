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
  extends Container[Tensor[T], Tensor[T], T]{

  private val internalChannel = nInputChannel * depthMultiplier

  val channelDim = if (dataFormat == DataFormat.NCHW) 2 else 4

  private val conv2Ds = (1 to nInputChannel).map(_ =>
    SpatialConvolution[T](
      nInputPlane = 1,
      nOutputPlane = depthMultiplier,
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
  ).toArray

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
    format = dataFormat
  )

  private val splitTable = SplitTable[T](channelDim).inputs()

  private val joinTable = JoinTable[T](channelDim, -1)

  private val graph = {
    val depthWiseOutputs = conv2Ds.zipWithIndex.map { case (n, i) =>
      n.inputs((splitTable, i + 1))
    }
    val join = joinTable.inputs(depthWiseOutputs)
    val outputNode = pointWiseConv2D.inputs(join)

    Graph[T](splitTable, outputNode)
  }

  graph.modules.foreach(this.add(_))

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "SpatialSeperableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeperableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")
    output = graph.forward(input).asInstanceOf[Tensor[T]]
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

    gradInput = graph.backward(input, gradOutput).asInstanceOf[Tensor[T]]
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

    gradInput = graph.updateGradInput(input, gradOutput).asInstanceOf[Tensor[T]]
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

    graph.accGradParameters(input, gradOutput).asInstanceOf[Tensor[T]]
  }
}

object SpatialSeperableConvolution {
  def apply[T: ClassTag](nInputChannel: Int, nOutputChannel: Int, depthMultiplier: Int,
    kW: Int, kH: Int, sW: Int, sH: Int, pW: Int, pH: Int, hasBias: Boolean, dataFormat: DataFormat,
    wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
    pRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T])
  : SpatialSeperableConvolution[T] = new SpatialSeperableConvolution(nInputChannel,
    nOutputChannel, depthMultiplier, kW, kH, sW, sH, pW, pH, hasBias, dataFormat, wRegularizer,
    bRegularizer)
}
