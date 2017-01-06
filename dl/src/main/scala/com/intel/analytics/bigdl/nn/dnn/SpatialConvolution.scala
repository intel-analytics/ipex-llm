/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.dnn

import com.intel.analytics.bigdl.mkl.{MKL, MklDnnFloat}
import com.intel.analytics.bigdl.nn.abstractnn.ModuleType._
import com.intel.analytics.bigdl.nn.{Default, InitializationMethod, Xavier}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.reflect.ClassTag

class SpatialConvolution[T: ClassTag](val nInputPlane: Int,
                                      val nOutputPlane: Int,
                                      val kW: Int,
                                      val kH: Int,
                                      val dW: Int = 1,
                                      val dH: Int = 1,
                                      val padW: Int = 0,
                                      val padH: Int = 0,
                                      val nGroup: Int = 1,
                                      val propagateBack: Boolean = true,
                                      private var initMethod: InitializationMethod = Default
                                     )(implicit ev: TensorNumeric[T]) extends MklModule[T] {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  class ConvRef extends Ref {
    var weight = new MklTensor[T]()
    var bias = new MklTensor[T]()

    var gradWeight = new MklTensor[T]()
    var gradBias = new MklTensor[T]()

    var weightInBwd = new MklTensor[T]()

    var gradOutputInBackWeight = new MklTensor[T]()
    var gradOutputInBackBias = new MklTensor[T]()
  }

  class ConvPrimitive extends Primitive {
    var backWeight = 0L
    var backBias = 0L
  }

  val refs = new ConvRef
  val primitive = new ConvPrimitive
  val resources = new Array[Long](ResourceType.dnnResourceNumber)

  // TODO currently, we should convert all weights, which maybe can be omited.
  val weight: Tensor[T] = Tensor[T](nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kH, kW)
  val gradWeight = Tensor[T]().resizeAs(weight)
  val bias: Tensor[T] = Tensor[T](nOutputPlane)
  val gradBias = Tensor[T](nOutputPlane)

  reset()

  private val im2colTime = 0L
  private val col2imTime = 0L

  def getIm2ColTime: Long = im2colTime
  def getCol2ImgTime: Long = col2imTime

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(kW * kH * nInputPlane)
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
      case Xavier =>
        val fanIn = nInputPlane * kH * kW
        val fanOut = nOutputPlane * kH * kW
        val stdv = math.sqrt(6.0 / (fanIn + fanOut))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
        bias.fill(ev.fromType(0))
      case _ => throw new IllegalArgumentException()
    }
  }

  private[this] def initLayerAttributes(input: Tensor[T]): Unit = {
    val strides = Array[Long](dW, dH)
    val pads = Array[Int](-padW, -padH)

    // set dimension = 4. because it seems that only support dimension 4 in mkl dnn
    val dimension = 4

    val inputLayout = new MklLayout(dimension, Utils.getSize(input, dimension))

    val outputLayout = new MklLayout(4, Array(
      Utils.computeOutput(inputLayout.size(0), padW, kW, dW), // width
      Utils.computeOutput(inputLayout.size(1), padH, kH, dH), // height
      nOutputPlane, // channels
      inputLayout.size(3)// number
    ))

    val kernelDim = if (nGroup == 1 || MKL.getMklVersion < 20160701) {
      dimension
    } else {
      dimension + 1
    }

    val nGroupMkl = if (MKL.getMklVersion < 20160701) {
      1
    } else {
      nGroup
    }

    val weightLayout = new MklLayout(kernelDim, Array(
      kW,
      kH,
      nInputPlane / nGroup,
      nOutputPlane / nGroupMkl,
      nGroupMkl
    ))

    val biasLayout = new MklLayout(1, Array(
      nOutputPlane
    ))

    for (tensor <- List(refs.weight, refs.gradWeight, refs.weightInBwd)) {
      tensor.resizeAs(weight)
    }

    for (tensor <- List(refs.bias, refs.gradBias)) {
      tensor.resizeAs(bias)
    }

    for (tensor <- List(refs.input, refs.gradInput)) {
      tensor.resizeAs(input)
    }

    for (tensor <- List(refs.output, refs.gradOutput, refs.gradOutputInBackWeight,
      refs.gradOutputInBackBias)) {
      tensor.resize(outputLayout.size.reverse.map(_.toInt),
        outputLayout.strides.reverse.map(_.toInt))
    }

    if (nextModuleType != DNN) {
      this.output.resizeAs(refs.output)
    }
    if (prevModuleType != DNN) {
      this.gradInput.resizeAs(refs.input)
    }

    initForward(inputLayout, outputLayout, weightLayout, strides, pads, biasLayout)
    initBackwardData(inputLayout, outputLayout, weightLayout, strides, pads, biasLayout)
    initBackwardWeight(inputLayout, outputLayout, weightLayout, strides, pads, biasLayout)
    initBackwardBias(outputLayout, biasLayout)

    setInit(true)
  }

  private[this] def initForward(inputLayout: MklLayout, outputLayout: MklLayout,
                                weightLayout: MklLayout,
                                strides: Array[Long], pads: Array[Int],
                                biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case FloatType =>
        this.primitive.forward = MklDnnFloat.convolutionCreateForward(1,
          nGroup,
          4,
          inputLayout.size,
          outputLayout.size,
          weightLayout.size,
          strides,
          pads,
          Border.dnnBorderZeros)
        require(this.primitive.forward != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.input.createConversion(inputLayout, primitive.forward, ResourceType.dnnResourceSrc)
    refs.weight.createConversion(weightLayout, primitive.forward, ResourceType.dnnResourceFilter)
    refs.bias.createConversion(biasLayout, primitive.forward, ResourceType.dnnResourceBias)
    refs.output.createConversion(outputLayout, primitive.forward, ResourceType.dnnResourceDst)
  }

  private[this] def initBackwardData(inputLayout: MklLayout, outputLayout: MklLayout,
                                     weightLayout: MklLayout,
                                     strides: Array[Long], pads: Array[Int],
                                     biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case FloatType =>
        this.primitive.backward = MklDnnFloat.convolutionCreateBackwardData(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          inputLayout.size,
          outputLayout.size,
          weightLayout.size,
          strides,
          pads,
          Border.dnnBorderZeros)
        require(this.primitive.backward != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.gradOutput.createConversion(outputLayout, primitive.backward,
      ResourceType.dnnResourceDiffDst)
    refs.weightInBwd.createConversion(weightLayout, primitive.backward,
      ResourceType.dnnResourceFilter)
    refs.gradInput.createConversion(inputLayout, primitive.backward,
      ResourceType.dnnResourceDiffSrc)
  }

  private[this] def initBackwardWeight(inputLayout: MklLayout, outputLayout: MklLayout,
                                       weightLayout: MklLayout,
                                       strides: Array[Long], pads: Array[Int],
                                       biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case FloatType =>
        this.primitive.backWeight = MklDnnFloat.convolutionCreateBackwardKernel(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          inputLayout.size,
          outputLayout.size,
          weightLayout.size,
          strides,
          pads,
          Border.dnnBorderZeros)
        require(this.primitive.backWeight != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.gradOutputInBackWeight.createConversion(outputLayout, primitive.backWeight,
      ResourceType.dnnResourceDiffDst)
    refs.gradWeight.createConversion(weightLayout, primitive.backWeight,
      ResourceType.dnnResourceDiffFilter)
  }

  private[this] def initBackwardBias(outputLayout: MklLayout, biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case FloatType =>
        this.primitive.backBias = MklDnnFloat.convolutionCreateBackwardBias(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          outputLayout.size)
        require(this.primitive.backBias != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    refs.gradOutputInBackBias.createConversion(outputLayout, primitive.backBias,
      ResourceType.dnnResourceDiffDst)
    refs.gradBias.createConversion(biasLayout, primitive.backBias, ResourceType.dnnResourceDiffBias)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!isInited) {
      initLayerAttributes(input)
    }

    refs.input.set(input)
    refs.weight.set(weight)
    refs.bias.set(bias)

    java.util.Arrays.fill(resources, 0)
    resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
    resources(ResourceType.dnnResourceBias) = refs.bias.getConvertedStorage()
    resources(ResourceType.dnnResourceFilter) = refs.weight.getConvertedStorage()
    resources(ResourceType.dnnResourceDst) = refs.output.mklStorage()

    execute(resources, primitive.forward)

    if (this.nextModuleType == DNN) {
      this.output = refs.output
    } else {
      refs.output.backToUsr(output)
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (propagateBack) {
      refs.weightInBwd.set(weight)
      refs.gradOutput.set(gradOutput)

      java.util.Arrays.fill(resources, 0)
      resources(ResourceType.dnnResourceDiffDst) = refs.gradOutput.getConvertedStorage()
      resources(ResourceType.dnnResourceFilter) = refs.weightInBwd.getConvertedStorage()
      resources(ResourceType.dnnResourceDiffSrc) = refs.gradInput.mklStorage()

      execute(resources, primitive.backward)

      if (this.prevModuleType == DNN) {
        this.gradInput = this.refs.gradInput
      } else {
        refs.gradInput.backToUsr(gradInput)
      }
    }

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T],
                                 gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    refs.input.set(input)
    refs.gradOutputInBackWeight.set(gradOutput)
    refs.gradOutputInBackBias.set(gradOutput)

    {
      java.util.Arrays.fill(resources, 0)
      resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
      resources(ResourceType.dnnResourceDiffDst) = refs.gradOutputInBackWeight.getConvertedStorage()
      resources(ResourceType.dnnResourceDiffFilter) = refs.gradWeight.mklStorage()

      execute(resources, primitive.backWeight)

      refs.gradWeight.backToUsr(gradWeight)
    }

    {
      java.util.Arrays.fill(resources, 0)
      resources(ResourceType.dnnResourceSrc) = refs.input.getConvertedStorage()
      resources(ResourceType.dnnResourceDiffDst) = refs.gradOutputInBackBias.getConvertedStorage()
      resources(ResourceType.dnnResourceDiffBias) = refs.gradBias.mklStorage()

      execute(resources, primitive.backBias)

      refs.gradBias.backToUsr(gradBias)
    }
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialConvolution[T]]) { return false }
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if (this.eq(other)) { return true }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      weight == other.weight &&
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString: String = {
    s"mkl.SpatialConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }
}
