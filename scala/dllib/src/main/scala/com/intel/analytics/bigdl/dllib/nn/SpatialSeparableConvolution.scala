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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, ModuleSerializer, SerializeContext}

import scala.reflect.ClassTag

/**
 * Separable convolutions consist in first performing a depthwise spatial convolution (which acts
 * on each input channel separately) followed by a pointwise convolution which mixes together the
 * resulting output channels. The  depthMultiplier argument controls how many output channels are
 * generated per input channel in the depthwise step.
 *
 * @param nInputChannel input image channel number
 * @param nOutputChannel output image channel number
 * @param depthMultiplier how many output channels are generated in the hidden depthwise step
 * @param kW kernel width
 * @param kH kernel height
 * @param sW stride width
 * @param sH stride height
 * @param pW padding width
 * @param pH padding height
 * @param hasBias do we use a bias on the output, default value is true
 * @param dataFormat image data format, which can be NHWC or NCHW, default value is NCHW
 * @param wRegularizer kernel parameter regularizer
 * @param bRegularizer bias regularizer
 * @param pRegularizer point wise kernel parameter regularizer
 * @param initDepthWeight kernel parameter init tensor
 * @param initPointWeight point wise kernel parameter init tensor
 * @param initBias bias init tensor
 * @tparam T module parameter numeric type
 */
class SpatialSeparableConvolution[T: ClassTag](
  val nInputChannel: Int,
  val nOutputChannel: Int,
  val depthMultiplier: Int,
  val kW: Int, val kH: Int,
  val sW: Int = 1, val sH: Int = 1,
  val pW: Int = 0, val pH: Int = 0,
  val hasBias: Boolean = true,
  val dataFormat: DataFormat = DataFormat.NCHW,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null,
  var pRegularizer: Regularizer[T] = null,
  val initDepthWeight: Tensor[T] = null,
  val initPointWeight: Tensor[T] = null,
  val initBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[T], T]{

  private val internalChannel = nInputChannel * depthMultiplier

  private val channelDim = if (dataFormat == DataFormat.NCHW) 2 else 4

  private val depthWeight = if (initDepthWeight != null) {
    initDepthWeight
  } else if (dataFormat == DataFormat.NCHW) {
    Tensor[T](depthMultiplier, nInputChannel, kW, kH)
  } else {
      Tensor[T](kW, kH, nInputChannel, depthMultiplier)
  }

  private val depthGradWeight = Tensor[T].resizeAs(depthWeight)

  private val pointWeight = if (initPointWeight != null) {
    initPointWeight
  } else if (dataFormat == DataFormat.NCHW) {
    Tensor[T](nOutputChannel, internalChannel, 1, 1)
  } else {
    Tensor[T](1, 1, internalChannel, nOutputChannel)
  }

  private val pointGradWeight = Tensor[T].resizeAs(pointWeight)

  private val bias = if (initBias != null) initBias else Tensor[T](nOutputChannel)

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

  reset()

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(depthWeight, pointWeight, bias), Array(depthGradWeight, pointGradWeight, gradBias))
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"SeparableConvolution2D requires 4D input, but got input dim ${input.length}")
    SpatialConvolution[T](nInputChannel, nOutputChannel, kW, kH,
      sW, sH, pW, pH, format = dataFormat).computeOutputShape(inputShape)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "SpatialSeparableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeparableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      s"input tensor channel dimension size(${input.size(channelDim)}) doesn't " +
        s"match layer nInputChannel $nInputChannel")

    SpatialSeparableConvolution.copyWeight(depthConv.weight, input.size(channelDim),
      depthMultiplier, depthWeight, dataFormat)

    depthConv.forward(input)
    output = pointWiseConv2D.forward(depthConv.output)
    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val before = System.nanoTime()
    require(input.nDimension() == 4, "SpatialSeparableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeparableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    require(gradOutput.nDimension() == 4, "SpatialSeparableConvolution only accept 4D gradOutput")
    require(gradOutput.isContiguous(), "SpatialSeparableConvolution require contiguous gradOutput")
    require(nOutputChannel == gradOutput.size(channelDim),
      "gradOutput tensor channel dimension size doesn't match layer nOutputChannel")

    pointWiseConv2D.backward(depthConv.output, gradOutput)
    gradInput = depthConv.backward(input, pointWiseConv2D.gradInput)
    SpatialSeparableConvolution.copyDepthGradWeight(nInputChannel, depthMultiplier,
      depthConv.gradWeight, depthGradWeight, dataFormat)
    backwardTime += System.nanoTime() - before
    gradInput
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "SpatialSeparableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeparableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    require(gradOutput.nDimension() == 4, "SpatialSeparableConvolution only accept 4D gradOutput")
    require(gradOutput.isContiguous(), "SpatialSeparableConvolution require contiguous gradOutput")
    require(nOutputChannel == gradOutput.size(channelDim),
      "gradOutput tensor channel dimension size doesn't match layer nOutputChannel")

    pointWiseConv2D.updateGradInput(depthConv.output, gradOutput)
    gradInput = depthConv.updateGradInput(input, pointWiseConv2D.gradInput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.nDimension() == 4, "SpatialSeparableConvolution only accept 4D input")
    require(input.isContiguous(), "SpatialSeparableConvolution require contiguous input")
    require(nInputChannel == input.size(channelDim),
      "input tensor channel dimension size doesn't match layer nInputChannel")

    require(gradOutput.nDimension() == 4, "SpatialSeparableConvolution only accept 4D gradOutput")
    require(gradOutput.isContiguous(), "SpatialSeparableConvolution require contiguous gradOutput")
    require(nOutputChannel == gradOutput.size(channelDim),
      "gradOutput tensor channel dimension size doesn't match layer nOutputChannel")

    pointWiseConv2D.accGradParameters(depthConv.output, gradOutput)
    depthConv.accGradParameters(input, pointWiseConv2D.gradInput)
    SpatialSeparableConvolution.copyDepthGradWeight(nInputChannel, depthMultiplier,
      depthConv.gradWeight, depthGradWeight, dataFormat)
  }

  override def reset(): Unit = {
    if (initDepthWeight == null) depthWeight.rand()
    if (initPointWeight == null) pointWeight.rand()
    if (initBias == null) bias.zero()
    zeroGradParameters()
  }
}

object SpatialSeparableConvolution extends ModuleSerializable {

  def apply[T: ClassTag](nInputChannel: Int, nOutputChannel: Int, depthMultiplier: Int,
    kW: Int, kH: Int, sW: Int = 1, sH: Int = 1, pW: Int = 0, pH: Int = 0,
    hasBias: Boolean = true, dataFormat: DataFormat = DataFormat.NCHW,
    wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
    pRegularizer: Regularizer[T] = null, initDepthWeight: Tensor[T] = null,
    initPointWeight: Tensor[T] = null, initBias: Tensor[T] = null
  )(implicit ev: TensorNumeric[T])
  : SpatialSeparableConvolution[T] = new SpatialSeparableConvolution[T](nInputChannel,
    nOutputChannel, depthMultiplier, kW, kH, sW, sH, pW, pH, hasBias, dataFormat, wRegularizer,
    bRegularizer, pRegularizer, initDepthWeight, initPointWeight, initBias)

  private[bigdl] def copyWeight[T](weight: Tensor[T], nInputChannel: Int,
    depthMultiplier: Int, sourceWeight: Tensor[T], dataFormat: DataFormat): Unit = {
    val kInputDim = if (dataFormat == DataFormat.NHWC) 3 else 2
    val kOutputDim = if (dataFormat == DataFormat.NHWC) 4 else 1
    val delta = if (dataFormat == DataFormat.NHWC) 0 else 1
    weight.zero()
    var in = 0
    while(in < nInputChannel) {
      var out = 0
      while(out < depthMultiplier) {
        // weight is a 5D tensor with a group dimension
        weight.select(kInputDim + 1, in + 1)
          .select(kOutputDim + delta, in * depthMultiplier + out + 1)
          .copy(sourceWeight.select(kInputDim, in + 1).select(kOutputDim - 1 + delta, out + 1))
        out += 1
      }
      in += 1
    }
  }

  private[bigdl] def copyDepthGradWeight[T](
    nInputChannel: Int, depthMultiplier: Int,
    sourceGrad: Tensor[T], targetGrad: Tensor[T], dataFormat: DataFormat
  ): Unit = {
    val kInputDim = if (dataFormat == DataFormat.NHWC) 3 else 2
    val kOutputDim = if (dataFormat == DataFormat.NHWC) 4 else 1
    val delta = if (dataFormat == DataFormat.NHWC) 0 else 1
    var in = 0
    while(in < nInputChannel) {
      var out = 0
      while(out < depthMultiplier) {
        targetGrad.select(kInputDim, in + 1).select(kOutputDim - 1 + delta, out + 1)
          .copy(sourceGrad.select(kInputDim + 1, in + 1).select(kOutputDim + delta,
            in * depthMultiplier + out + 1))
        out += 1
      }
      in += 1
    }
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val ssc = super.doLoadModule(context).asInstanceOf[SpatialSeparableConvolution[T]]
    val weights = ssc.parameters()._1
    val (depthWeight, pointWeight, bias) = (weights(0), weights(1), weights(2))

    val depthWeightLoad = DataConverter.
      getAttributeValue(context, attrMap.get("depthWeight")).
      asInstanceOf[Tensor[T]]
    depthWeight.copy(depthWeightLoad)

    val pointWeightLoad = DataConverter.
      getAttributeValue(context, attrMap.get("pointWeight")).
      asInstanceOf[Tensor[T]]
    pointWeight.copy(pointWeightLoad)

    val biasLoad = DataConverter.
      getAttributeValue(context, attrMap.get("bias")).
      asInstanceOf[Tensor[T]]
    bias.copy(biasLoad)

    ssc.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    sreluBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, sreluBuilder)

    val ssc = context.moduleData.module.asInstanceOf[SpatialSeparableConvolution[T]]
    val weights = ssc.parameters()._1
    val (depthWeight, pointWeight, bias) = (weights(0), weights(1), weights(2))

    val depthWeightBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, depthWeightBuilder,
      depthWeight, ModuleSerializer.tensorType)
    sreluBuilder.putAttr("depthWeight", depthWeightBuilder.build)

    val pointWeightBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, pointWeightBuilder,
      pointWeight, ModuleSerializer.tensorType)
    sreluBuilder.putAttr("pointWeight", pointWeightBuilder.build)

    val biasBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, biasBuilder,
      bias, ModuleSerializer.tensorType)
    sreluBuilder.putAttr("bias", biasBuilder.build)
  }
}
