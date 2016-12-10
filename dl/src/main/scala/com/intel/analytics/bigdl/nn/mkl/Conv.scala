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

package com.intel.analytics.bigdl.nn.mkl

import com.intel.analytics.bigdl.mkl.{MKL, MklDnnFloat}
import com.intel.analytics.bigdl.nn.{InitializationMethod, Default, Xavier, Constant}
import com.intel.analytics.bigdl.nn.ModuleType._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.language.implicitConversions
import scala.reflect.ClassTag

class Conv[@specialized(Float, Double) T: ClassTag](
    val nInputPlane: Int,
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
)(implicit ev: TensorNumeric[T])
    extends MklModule[T] {

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

  // TODO currently, we should convert all weights, which maybe can be omited.
  val weight: Tensor[T] =
  Tensor[T](nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kH, kW)
  var gradWeight = Tensor[T]().resizeAs(weight)
  var bias: Tensor[T] = Tensor[T](nOutputPlane)
  var gradBias = Tensor[T](nOutputPlane)

//  reset()

  private var im2colTime = 0L
  private var col2imTime = 0L

  def getIm2ColTime(): Long = im2colTime
  def getCol2ImgTime(): Long = col2imTime

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
      case Constant =>
        weight.fill(ev.fromType(0.1))
        bias.fill(ev.fromType(0))
    }
  }

  private def initLayerAttributes(input: Tensor[T]): Unit = {
    val strides = Array[Long](dW, dH)
    val pads = Array[Int](-padW, -padH)

    // set dimension = 4 forcily. because it seems that only support dimension 4 in mkl dnn
    val dimension = 4

    val inputLayout = new MklLayout(4, Array(
      input.size(input.dim()), // width
      input.size(input.dim() - 1), // height
      if (input.dim() < 3) 1 else input.size(input.dim() - 2), // channels
      if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number
    ))

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
      nGroupMkl))

    val biasLayout = new MklLayout(1, Array(
      nOutputPlane
    ))

    refs.input.resizeAs(input)
    refs.gradInput.resizeAs(input)

//    bias.resize(biasLayout.size.map(_.toInt),
//      biasLayout.strides.reverse.map(_.toInt))
//    weight.resize(weightLayout.size.reverse.map(_.toInt),
//      weightLayout.strides.reverse.map(_.toInt))
//    gradBias.resizeAs(bias)
//    gradWeight.resizeAs(weight)

    refs.weight.resizeAs(weight)
    refs.bias.resizeAs(bias)
    refs.gradBias.resizeAs(bias)
    refs.gradWeight.resizeAs(weight)
    refs.weightInBwd.resizeAs(weight)

    refs.output.resize(outputLayout.size.reverse.map(_.toInt),
      outputLayout.strides.reverse.map(_.toInt))
    refs.gradOutput.resizeAs(refs.output)
    refs.gradOutputInBackBias.resizeAs(refs.gradOutput)
    refs.gradOutputInBackWeight.resizeAs(refs.gradOutput)

    this.output.resizeAs(refs.output)
    this.gradInput.resizeAs(refs.input)

    initForward(inputLayout, outputLayout, weightLayout, strides, pads, biasLayout)
    initBackwardData(inputLayout, outputLayout, weightLayout, strides, pads, biasLayout)
    initBackwardWeight(inputLayout, outputLayout, weightLayout, strides, pads, biasLayout)
    initBackwardBias(outputLayout, biasLayout)

    isInited_=(true)
  }

  private def initForward(inputLayout: MklLayout, outputLayout: MklLayout,
                          weightLayout: MklLayout,
                          strides: Array[Long], pads: Array[Int],
                          biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case "Float" =>
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

  private def initBackwardData(inputLayout: MklLayout, outputLayout: MklLayout,
                               weightLayout: MklLayout,
                               strides: Array[Long], pads: Array[Int],
                               biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case "Float" =>
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

    refs.gradOutput.createConversion(outputLayout, primitive.backward, ResourceType.dnnResourceDiffDst)
    refs.weightInBwd.createConversion(weightLayout, primitive.backward, ResourceType.dnnResourceFilter)
    refs.gradInput.createConversion(inputLayout, primitive.backward, ResourceType.dnnResourceDiffSrc)
  }

  def initBackwardWeight(inputLayout: MklLayout, outputLayout: MklLayout,
                         weightLayout: MklLayout,
                         strides: Array[Long], pads: Array[Int],
                         biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case "Float" =>
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
    // we do not create input mkl again TODO
  }

  def initBackwardBias(outputLayout: MklLayout, biasLayout: MklLayout): Unit = {
    ev.getType() match {
      case "Float" =>
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

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.convolutionForwardExecute(
          refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.weight.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.bias.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.output.mklStorage().array().asInstanceOf[Array[Float]],
          primitive.forward
        )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (this.nextModuleType() == DNN) {
      this.output = refs.output
    } else {
      refs.output.backToUsr(output.storage(), output.storageOffset())
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (propagateBack) {
      // weightInBwd here converted from weight which converted at updateOutput
      refs.weightInBwd.set(weight)
      refs.gradOutput.set(gradOutput)

      ev.getType() match {
        case "Float" =>
          MklDnnFloat.convolutionBackwardDataExecute(
            refs.gradInput.mklStorage().array().asInstanceOf[Array[Float]],
            refs.gradOutput.getConvertedStorage().array().asInstanceOf[Array[Float]],
            refs.weightInBwd.getConvertedStorage().array().asInstanceOf[Array[Float]],
            primitive.backward)
      }

      if (this.prevModuleType() == DNN) {
        this.gradInput = this.refs.gradInput
      } else {
        refs.gradInput.backToUsr(this.gradInput.storage(), this.gradInput.storageOffset())
      }
      println(getName())
    }

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T],
                                 gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    refs.input.set(input)
    refs.gradOutputInBackWeight.set(gradOutput)
    refs.gradOutputInBackBias.set(gradOutput)

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.convolutionBackwardKernelExecute(
          refs.input.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.gradOutputInBackWeight.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.gradWeight.mklStorage().array().asInstanceOf[Array[Float]],
          primitive.backWeight
        )
        MklDnnFloat.convolutionBackwardBiasExecute(
          refs.gradOutputInBackBias.getConvertedStorage().array().asInstanceOf[Array[Float]],
          refs.gradBias.mklStorage().array().asInstanceOf[Array[Float]],
          primitive.backBias
        )
    }

    refs.gradWeight.backToUsr(gradWeight.storage(), gradWeight.storageOffset())
    refs.gradBias.backToUsr(gradBias.storage(), gradBias.storageOffset())
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
      kW == other.kernelWidth &&
      kH == other.kernelHeight &&
      dW == other.strideWidth &&
      dH == other.strideHeight &&
      padW == other.padWidth &&
      padH == other.padHeight &&
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

  override def toString(): String = {
    s"mkl.SpatialConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }
}
