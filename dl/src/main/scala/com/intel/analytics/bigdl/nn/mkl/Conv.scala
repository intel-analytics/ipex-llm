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

import java.awt.Dimension

import com.intel.analytics.bigdl.mkl.{MKL, MklDnnDouble, MklDnnFloat}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

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

  val weight = new MklTensor[T]()
  val bias = new MklTensor[T]()

  val gradWeight = Tensor()
  val gradBias = Tensor()

  class ConvRef extends Ref {
    var weight = new MklTensor[T]()
    var bias = new MklTensor[T]()

    var gradWeight = new MklTensor[T]()
    var gradBias = new MklTensor[T]()
  }

  val refs = new ConvRef

  val gradWeightMkl = new MklTensor[T]()
  val gradBiasMkl = new MklTensor[T]()

  val weightInBackData: MklTensor[T] = new MklTensor[T]()

  var gradOutputInBackWeight = new MklTensor[T]()
  var gradOutputInBackBias = new MklTensor[T]()

  var backWeightPrim = 0L
  var backBiasPrim = 0L

  var firstPassBackParam = true

  private def initLayerAttributes(input: Tensor[T]): Unit = {
    val weightSize = new Array[Long](5)
    val weightStrides = new Array[Long](5)

    val biasSize = new Array[Long](1)
    val biasStrides = new Array[Long](1)

    val strides = Array[Long](dW, dH)
    val pads = Array[Int](-padW, -padH)
    var dimension = 0
    var kernelDim = 0

    dimension = input.dim()

    inputSize(0) = input.size(input.dim()) // width
    inputSize(1) = input.size(input.dim() - 1) // height
    inputSize(2) = if (input.dim() < 3) 1 else input.size(input.dim() - 2) // channels
    inputSize(3) = if (input.dim() < 4) 1 else input.size(input.dim() - 3) // number

    require(inputSize(2) == nInputPlane, "nInputPlane should be equal to inputSize(2)")

    outputSize(0) = Utils.computeOutput(inputSize(0), padW, kW, dW) // width
    outputSize(1) = Utils.computeOutput(inputSize(1), padH, kH, dH) // height
    outputSize(2) = nOutputPlane // channels
    outputSize(3) = inputSize(3) // number

    kernelDim = if (nGroup == 1 || MKL.getMklVersion < 20160701) {
      dimension
    } else {
      dimension + 1
    }

    val nGroupMkl = if (MKL.getMklVersion < 20160701) {
      1
    } else {
      nGroup
    }

    weightSize(0) = kW
    weightSize(1) = kH
    weightSize(2) = nInputPlane / nGroup
    weightSize(3) = nOutputPlane / nGroupMkl
    weightSize(4) = nGroupMkl

    biasSize(0) = nOutputPlane
    biasStrides(0) = 1

    def computeStrides(size: Array[Long], stride: Array[Long]): Unit = {
      stride(0) = 1
      for (i <- 1 until size.length) {
        stride(i) = size(i - 1) * stride(i - 1)
      }
    }

    computeStrides(inputSize, inputStrides)
    computeStrides(outputSize, outputStrides)
    computeStrides(weightSize, weightStrides)

    inputMkl.resizeAs(input)
    gradInputMkl.resizeAs(input)

    bias.resize(biasSize.map(_.toInt), biasStrides.reverse.map(_.toInt))
    weight.resize(weightSize.reverse.map(_.toInt), weightStrides.reverse.map(_.toInt))
    gradBias.resizeAs(bias)
    gradWeight.resizeAs(weight)
    gradBiasMkl.resizeAs(bias)
    gradWeightMkl.resizeAs(weight)
    weightInBackData.resizeAs(weight)

    outputMkl.resize(outputSize.reverse.map(_.toInt), outputStrides.reverse.map(_.toInt))
    gradOutputMkl.resizeAs(outputMkl)
    gradOutputInBackBias.resizeAs(gradOutputMkl)
    gradOutputInBackWeight.resizeAs(gradOutputMkl)

    initForward(dimension, kernelDim, weightSize, weightStrides,
      strides, pads, biasSize, biasStrides)
    initBackwardData(dimension, kernelDim, weightSize, weightStrides,
      strides, pads, biasSize, biasStrides)
    initBackwardWeight(dimension, kernelDim, weightSize, weightStrides,
      strides, pads, biasSize, biasStrides)
    initBackwardBias(dimension, biasSize, biasStrides)
  }

  private def initForward(dimension: Int, kernelDim: Int,
                          weightSize: Array[Long], weightStrides: Array[Long],
                          strides: Array[Long], pads: Array[Int],
                          biasSize: Array[Long], biasStrides: Array[Long]): Unit = {
    ev.getType() match {
      case "Float" =>
        this.forwardPrim = MklDnnFloat.convolutionCreateForward(1,
                                                                nGroup,
                                                                4,
                                                                inputSize,
                                                                outputSize,
                                                                weightSize,
                                                                strides,
                                                                pads,
                                                                Border.dnnBorderZeros)
        require(this.forwardPrim != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    inputMkl.createConversion(dimension, inputSize, inputStrides, forwardPrim,
      ResourceType.dnnResourceSrc, MklRWType.READ)
    weight.createConversion(kernelDim, weightSize, weightStrides, forwardPrim,
      ResourceType.dnnResourceFilter, MklRWType.READ)
    bias.createConversion(1, biasSize, biasStrides, forwardPrim,
      ResourceType.dnnResourceBias, MklRWType.READ)
    outputMkl.createConversion(dimension, outputSize, outputStrides, forwardPrim,
      ResourceType.dnnResourceDst, MklRWType.WRITE)
  }

  private def initBackwardData(dimension: Int, kernelDim: Int,
                               weightSize: Array[Long], weightStrides: Array[Long],
                               strides: Array[Long], pads: Array[Int],
                               biasSize: Array[Long], biasStrides: Array[Long]): Unit = {
    ev.getType() match {
      case "Float" =>
        this.backwardPrim = MklDnnFloat.convolutionCreateBackwardData(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          inputSize,
          outputSize,
          weightSize,
          strides,
          pads,
          Border.dnnBorderZeros)
        require(this.backwardPrim != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    gradOutputMkl.createConversion(dimension, outputSize, outputStrides, backwardPrim,
      ResourceType.dnnResourceDiffDst, MklRWType.READ)
    weightInBackData.createConversion(kernelDim, weightSize, weightStrides,
      backwardPrim, ResourceType.dnnResourceFilter, MklRWType.READ)
    gradInputMkl.createConversion(dimension, inputSize, inputStrides, backwardPrim,
      ResourceType.dnnResourceDiffSrc, MklRWType.WRITE)
  }

  def initBackwardWeight(dimension: Int, kernelDim: Int,
                     weightSize: Array[Long], weightStrides: Array[Long],
                     strides: Array[Long], pads: Array[Int],
                     biasSize: Array[Long], biasStrides: Array[Long]): Unit = {
    ev.getType() match {
      case "Float" =>
        this.backWeightPrim = MklDnnFloat.convolutionCreateBackwardKernel(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          inputSize,
          outputSize,
          weightSize,
          strides,
          pads,
          Border.dnnBorderZeros)
        require(this.backWeightPrim != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    gradOutputInBackWeight.createConversion(dimension, outputSize, outputStrides, backWeightPrim,
      ResourceType.dnnResourceDiffDst, MklRWType.READ)
//    gradWeight.createInterLayout(backWeightPrim, ResourceType.dnnResourceDiffFilter)
    gradWeightMkl.createConversion(kernelDim, weightSize, weightStrides, backWeightPrim,
      ResourceType.dnnResourceDiffFilter, MklRWType.WRITE)
  }

  def initBackwardBias(dimension: Int, biasSize: Array[Long], biasStrides: Array[Long]): Unit = {
    ev.getType() match {
      case "Float" =>
        this.backBiasPrim = MklDnnFloat.convolutionCreateBackwardBias(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          outputSize)
        require(this.backBiasPrim != 0, "create convolution primitive failed.")
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    gradOutputInBackBias.createConversion(dimension, outputSize, outputStrides, backBiasPrim,
      ResourceType.dnnResourceDiffDst, MklRWType.READ)
//    gradBias.createInterLayout(backBiasPrim, ResourceType.dnnResourceDiffBias)
    gradBiasMkl.createConversion(1, biasSize, biasStrides, backBiasPrim,
      ResourceType.dnnResourceDiffBias, MklRWType.WRITE)
  }

  override def convertToMklDnn(input: Tensor[T]): Unit = {
    initLayerAttributes(input)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (input.isMklTensor()) {
      inputMkl = input.asInstanceOf[MklTensor[T]]
    }

//    inputMkl.set(input)

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.convolutionForwardExecute(
          inputMkl.getConvertedStorage(input).array().asInstanceOf[Array[Float]],
          weight.storage.array().asInstanceOf[Array[Float]],
          bias.storage.array().asInstanceOf[Array[Float]],
          outputMkl.getStroage().array().asInstanceOf[Array[Float]],
          forwardPrim
        )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (this.getNextPtr() != 0) {
      this.output = outputMkl
    } else {
      outputMkl.backToUsr(output, ConvertType.INTERNALTOUSR)
    }

    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradOutput.isMklTensor()) {
      gradOutputMkl = gradOutput.asInstanceOf[MklTensor[T]]
    }

    val tmp = Tensor().resizeAs(weight)
    tmp.storage().set(weight.getUsrStorage())

//    MklTensor.convert(weight.usrStorage, weight.usrOffset - 1,
//      weightInBackData.storage(), weightInBackData.usrToMkl, toMkl = true)
//    weightInBackData.set(weight)
    ev.getType() match {
      case "Float" =>
        MklDnnFloat.convolutionBackwardDataExecute(
          gradInputMkl.getStroage().array().asInstanceOf[Array[Float]],
          gradOutputMkl.getConvertedStorage(gradOutput).array().asInstanceOf[Array[Float]],
          weightInBackData.getConvertedStorage(tmp).array().asInstanceOf[Array[Float]],
          backwardPrim)
    }

    if (this.getPrevPtr() != 0) {
      this.gradInput = this.gradInputMkl
    } else {
      gradInputMkl.backToUsr(this.gradInput, ConvertType.INTERNALTOUSR)
    }

    val weightTmp = Tensor[T]().resizeAs(weight)
    weight.backToUsr(weightTmp, ConvertType.MKLTOUSR)

    val gradOutputTmp = Tensor[T]().resizeAs(gradOutput)
    gradOutputMkl.backToUsr(gradOutputTmp, ConvertType.MKLTOUSR)

    this.gradInput
  }

  override def accGradParameters(input: Tensor[T],
                                 gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    if (gradOutput.isMklTensor()) {
      gradOutputInBackWeight = gradOutput.asInstanceOf[MklTensor[T]]
      gradOutputInBackBias = gradOutput.asInstanceOf[MklTensor[T]]
    }

    if (input.isMklTensor()) {
      inputMkl = input.asInstanceOf[MklTensor[T]]
    }

    ev.getType() match {
      case "Float" =>
        MklDnnFloat.convolutionBackwardKernelExecute(
          inputMkl.getConvertedStorage(input).array().asInstanceOf[Array[Float]],
          gradOutputInBackWeight.getConvertedStorage(gradOutput).array().asInstanceOf[Array[Float]],
          gradWeightMkl.getStroage().array().asInstanceOf[Array[Float]],
          backWeightPrim
        )
        MklDnnFloat.convolutionBackwardBiasExecute(
          gradOutputInBackBias.getConvertedStorage(gradOutput).array().asInstanceOf[Array[Float]],
          gradBiasMkl.getStroage().array().asInstanceOf[Array[Float]],
          backBiasPrim
        )
    }

    gradWeightMkl.backToUsr(gradWeight, ConvertType.INTERNALTOUSR)
    gradBiasMkl.backToUsr(gradBias, ConvertType.INTERNALTOUSR)
  }

  override def updateParameters(learningRate: T): Unit = {}

  override def zeroGradParameters(): Unit = {}

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array[Tensor[T]](Tensor[T]), Array[Tensor[T]](Tensor[T]))
  }

  override def equals(obj: Any): Boolean = { true }

  override def hashCode(): Int = { 1 }

  override def toString(): String = {
    s"mkl.SpatialConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }
}
