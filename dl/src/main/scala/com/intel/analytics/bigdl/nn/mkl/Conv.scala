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

  var inputMkl = new MklTensor[T]()
  var gradOutputMkl = new MklTensor[T]()

  val weight: MklTensor[T] = new MklTensor[T]().resize(nGroup, nOutputPlane / nGroup,
    nInputPlane / nGroup, kH, kW).asInstanceOf[MklTensor[T]]
  val bias = new MklTensor[T]().resize(nOutputPlane).asInstanceOf[MklTensor[T]]

  val gradWeight = new MklTensor[T]().resizeAs(weight).asInstanceOf[MklTensor[T]]
  val gradBias = new MklTensor[T]().resizeAs(bias).asInstanceOf[MklTensor[T]]

  var outputMkl = new MklTensor[T]()
  var gradInputMkl = new MklTensor[T]()

  val backWeight: MklTensor[T] = new MklTensor[T]().resizeAs(weight)
    .set(weight)
    .asInstanceOf[MklTensor[T]]

  var gradOutputWeight = new MklTensor[T]()
  var gradOutputBias = new MklTensor[T]()

  var backWeightPrim = 0L
  var backBiasPrim = 0L

  val weightSize = new Array[Long](5)
  val weightStrides = new Array[Long](5)

  val biasSize = new Array[Long](1)
  val biasStrides = new Array[Long](1)

  val strides = Array[Long](dW, dH)
  val pads = Array[Int](-padW, -padH)
  var dimension = 0
  var kernelDim = 0

  var firstPassBackWeight = true

  private def initLayerAttributes(input: Tensor[T]): Unit = {
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
  }

  private def initForward(input: Tensor[T], output: Tensor[T]): Unit = {
    initLayerAttributes(input)
    output.resize(outputSize.reverse.map(_.toInt))

    ev.getType() match {
      case "Double" =>
        this.forwardPrim = MklDnnDouble.convolutionCreateForward()
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
    }

    input.asInstanceOf[MklTensor[T]].createUsrLayout(dimension, inputSize, inputStrides)
    input.asInstanceOf[MklTensor[T]].createMklLayout(forwardPrim, ResourceType.dnnResourceSrc)

    output.asInstanceOf[MklTensor[T]].createUsrLayout(dimension, outputSize, outputStrides)
    output.asInstanceOf[MklTensor[T]].createMklLayout(forwardPrim, ResourceType.dnnResourceDst)

    weight.asInstanceOf[MklTensor[T]].createUsrLayout(kernelDim, weightSize, weightStrides)
    weight.asInstanceOf[MklTensor[T]].createMklLayout(forwardPrim, ResourceType.dnnResourceFilter)

    bias.asInstanceOf[MklTensor[T]].createUsrLayout(1, biasSize, biasStrides)
    bias.asInstanceOf[MklTensor[T]].createMklLayout(forwardPrim, ResourceType.dnnResourceBias)
  }

  private def initBackward(input: MklTensor[T],
                           gradOutput: MklTensor[T],
                           gradInput: MklTensor[T]): Unit = {
    initLayerAttributes(input)
    gradInput.resize(inputSize.reverse.map(_.toInt))

    ev.getType() match {
      case "Double" =>
        this.backwardPrim = MklDnnDouble.convolutionCreateBackwardData()
        this.backWeightPrim = MklDnnDouble.convolutionCreateBackwardKernel()
        this.backBiasPrim = MklDnnDouble.convolutionCreateBackwardBias()
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
    }

    gradOutput.createUsrLayout(dimension, outputSize, outputStrides)
    gradOutput.createMklLayout(backwardPrim, ResourceType.dnnResourceDiffDst)

    gradInput.createUsrLayout(dimension, inputSize, inputStrides)
    gradInput.createMklLayout(backwardPrim, ResourceType.dnnResourceDiffSrc)

    backWeight.createUsrLayout(kernelDim, weightSize, weightStrides)
    backWeight.createMklLayout(backwardPrim, ResourceType.dnnResourceFilter)
  }

  def initBackWeight(input: MklTensor[T], gradOutput: MklTensor[T]): Unit = {
    initLayerAttributes(input)
    gradWeight.resizeAs(weight)

    ev.getType() match {
      case "Double" =>
        this.backwardPrim = MklDnnDouble.convolutionCreateBackwardData()
        this.backWeightPrim = MklDnnDouble.convolutionCreateBackwardKernel()
        this.backBiasPrim = MklDnnDouble.convolutionCreateBackwardBias()
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
        require(this.backwardPrim != 0, "create convolution primitive failed.")
    }

    gradOutput.createUsrLayout(dimension, outputSize, outputStrides)
    gradOutput.createMklLayout(backWeightPrim, ResourceType.dnnResourceDiffDst)

//    input.createUsrLayout(dimension, inputSize, inputStrides)
//    input.createMklLayout(backWeightPrim, ResourceType.dnnResourceSrc)

    gradWeight.createUsrLayout(kernelDim, weightSize, weightStrides)
    gradWeight.createMklLayout(backWeightPrim, ResourceType.dnnResourceDiffFilter)
  }

  def initBackBias(input: MklTensor[T], gradOutput: MklTensor[T]): Unit = {
    initLayerAttributes(input)
    gradBias.resizeAs(bias)

    ev.getType() match {
      case "Double" =>
        this.backwardPrim = MklDnnDouble.convolutionCreateBackwardData()
        this.backBiasPrim = MklDnnDouble.convolutionCreateBackwardKernel()
        this.backBiasPrim = MklDnnDouble.convolutionCreateBackwardBias()
      case "Float" =>
        this.backBiasPrim = MklDnnFloat.convolutionCreateBackwardBias(
          Algorithm.dnnAlgorithmConvolutionDirect,
          nGroup,
          4,
          outputSize)
        require(this.backBiasPrim != 0, "create convolution primitive failed.")
    }

    gradOutput.createUsrLayout(dimension, outputSize, outputStrides)
    gradOutput.createMklLayout(backBiasPrim, ResourceType.dnnResourceDiffDst)

    gradBias.createUsrLayout(1, biasSize, biasStrides)
    gradBias.createMklLayout(backBiasPrim, ResourceType.dnnResourceDiffBias)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (input.isMklTensor()) {
      inputMkl = input.asInstanceOf[MklTensor[T]]
    } else {
      inputMkl.resizeAs(input).set(input)
    }

    if (firstPassForward) {
      initForward(inputMkl, outputMkl)
      firstPassForward = false
    }

    inputMkl.convert(toMkl = true)
    weight.asInstanceOf[MklTensor[T]].convert(toMkl = true)
    bias.asInstanceOf[MklTensor[T]].convert(toMkl = true)

    ev.getType() match {
      case "Double" =>
        MklDnnDouble.convolutionForwardExecute(
          )
      case "Float" =>
        MklDnnFloat.convolutionForwardExecute(
          inputMkl.storageMkl.array().asInstanceOf[Array[Float]],
          weight.asInstanceOf[MklTensor[T]].storageMkl.array().asInstanceOf[Array[Float]],
          bias.asInstanceOf[MklTensor[T]].storageMkl.array().asInstanceOf[Array[Float]],
          outputMkl.storageMkl.array().asInstanceOf[Array[Float]],
          forwardPrim
        )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }

    if (this.getNextPtr() == 0) {
      this.outputMkl.convert(toMkl = false)
    }

    this.output = outputMkl
    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradOutput.isMklTensor()) {
      gradOutputMkl = gradOutput.asInstanceOf[MklTensor[T]]
    } else {
      gradOutputMkl.resizeAs(gradOutput).set(gradOutput)
    }

    if (firstPassBackward) {
      initBackward(inputMkl, gradOutputMkl, gradInputMkl)
      firstPassBackward_=(false)
    }

    gradOutputMkl.convert(toMkl = true)
    backWeight.set(weight)
    backWeight.convert(toMkl = true)

    ev.getType() match {
      case "Double" => MklDnnDouble.convolutionBackwardExecute()
      case "Float" =>
        MklDnnFloat.convolutionBackwardDataExecute(
          gradInputMkl.storageMkl.array().asInstanceOf[Array[Float]],
          gradOutputMkl.storageMkl.array().asInstanceOf[Array[Float]],
          backWeight.storageMkl.array().asInstanceOf[Array[Float]],
          backwardPrim)
    }

    if (this.getPrevPtr() == 0) {
      this.gradInputMkl.convert(toMkl = false)
    }

    this.gradInput = this.gradInputMkl
    this.gradInput
  }

  override def accGradParameters(input: Tensor[T],
                                 gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    if (gradOutput.isMklTensor()) {
      gradOutputWeight = gradOutput.asInstanceOf[MklTensor[T]]
      gradOutputBias = gradOutput.asInstanceOf[MklTensor[T]]
    } else {
      gradOutputWeight.resizeAs(gradOutput).set(gradOutput)
      gradOutputBias.resizeAs(gradOutput).set(gradOutput)
    }

    if (input.isMklTensor()) {
      inputMkl = input.asInstanceOf[MklTensor[T]]
    } else {
      inputMkl.resizeAs(input).set(input)
    }

    if (firstPassBackWeight) {
      initBackWeight(inputMkl, gradOutputWeight)
      initBackBias(inputMkl, gradOutputBias)
      firstPassBackWeight = false
    }

    gradOutputWeight.convert(toMkl = true)
    gradOutputBias.convert(toMkl = true)
    inputMkl.convert(toMkl = true)
    ev.getType() match {
      case "Double" => MklDnnDouble.convolutionCreateBackwardKernel()
        MklDnnDouble.convolutionCreateBackwardBias()
      case "Float" => MklDnnFloat.convolutionBackwardKernelExecute(
        inputMkl.storageMkl.array().asInstanceOf[Array[Float]],
        gradOutputWeight.storageMkl.array().asInstanceOf[Array[Float]],
        gradWeight.storageMkl.array().asInstanceOf[Array[Float]],
        backWeightPrim
      )
        MklDnnFloat.convolutionBackwardBiasExecute(
          gradOutputBias.storageMkl.array().asInstanceOf[Array[Float]],
          gradBias.storageMkl.array().asInstanceOf[Array[Float]],
          backBiasPrim
        )
    }

    gradWeight.convert(toMkl = false)
    gradBias.convert(toMkl = false)

    println("")
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
