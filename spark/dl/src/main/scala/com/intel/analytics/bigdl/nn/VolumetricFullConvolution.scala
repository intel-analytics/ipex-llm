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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{DenseTensorBLAS, DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Apply a 3D full convolution over an 3D input image, a sequence of images, or a video etc.
 * The input tensor is expected to be a 4D or 5D(with batch) tensor. Note that instead
 * of setting adjT, adjW and adjH, [[VolumetricConvolution]] also accepts a table input
 * with two tensors: T(convInput, sizeTensor) where convInput is the standard input tensor,
 * and the size of sizeTensor is used to set the size of the output (will ignore the adjT, adjW and
 * adjH values used to construct the module). This module can be used without a bias by setting
 * parameter noBias = true while constructing the module.
 *
 * If input is a 4D tensor nInputPlane x depth x height x width,
 * odepth  = (depth  - 1) * dT - 2*padT + kT + adjT
 * owidth  = (width  - 1) * dW - 2*padW + kW + adjW
 * oheight = (height - 1) * dH - 2*padH + kH + adjH
 *
 * Other frameworks call this operation "In-network Upsampling", "Fractionally-strided convolution",
 * "Backwards Convolution," "Deconvolution", or "Upconvolution."
 *
 * Reference Paper: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic
 * segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
 * 2015: 3431-3440.
 *
 * @param nInputPlane The number of expected input planes in the image given into forward()
 * @param nOutputPlane The number of output planes the convolution layer will produce.
 * @param kT The kernel depth of the convolution.
 * @param kW The kernel width of the convolution.
 * @param kH The kernel height of the convolution.
 * @param dT The step of the convolution in the depth dimension. Default is 1.
 * @param dW The step of the convolution in the width dimension. Default is 1.
 * @param dH The step of the convolution in the height dimension. Default is 1.
 * @param padT The additional zeros added per depth to the input planes. Default is 0.
 * @param padW The additional zeros added per width to the input planes. Default is 0.
 * @param padH The additional zeros added per height to the input planes. Default is 0.
 * @param adjT Extra depth to add to the output image. Default is 0.
 * @param adjW Extra width to add to the output image. Default is 0.
 * @param adjH Extra height to add to the output image. Default is 0.
 * @param nGroup Kernel group number.
 * @param noBias If bias is needed.
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */

@SerialVersionUID(- 809921720980508072L)
class VolumetricFullConvolution[T: ClassTag](
  val nInputPlane: Int,
  val nOutputPlane: Int,
  val kT: Int,
  val kW: Int,
  val kH: Int,
  val dT: Int = 1,
  val dW: Int = 1,
  val dH: Int = 1,
  val padT: Int = 0,
  val padW: Int = 0,
  val padH: Int = 0,
  var adjT: Int = 0,
  var adjW: Int = 0,
  var adjH: Int = 0,
  val nGroup: Int = 1,
  val noBias: Boolean = false,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Tensor[T], T] with Initializable {

  require(adjW <= dW - 1 && adjH <= dH - 1 && adjT <= dT -1,
    s"VolumetricFullConvolution: adjW=$adjW and adjH=$adjH must be smaller than " +
      s"(dW - 1)=${dW - 1} and (dH - 1)=${dH - 1} respectively")

  val weight: Tensor[T] = Tensor[T](nGroup, nInputPlane / nGroup,
    nOutputPlane / nGroup, kT, kH, kW)
  val bias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)

  val gradWeight: Tensor[T] = Tensor[T](nGroup, nInputPlane / nGroup,
    nOutputPlane / nGroup, kT, kH, kW)
  val gradBias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)
  private val columns: Tensor[T] = Tensor[T]()
  private val ones: Tensor[T] = Tensor[T]()
  private val zeroScalar: Tensor[T] = Tensor[T]()
  protected val onesBias = if (noBias) null else Tensor[T]()
  protected val onesBatch = Tensor[T]()
  protected var weightMM: Tensor[T] = _
  protected val gradientBiasMT: Tensor[T] = if (noBias) null else Tensor[T]()
  protected val gradWeightMMInBatch: Tensor[T] = Tensor[T]()

  protected val _1x1x1 = if (
    kH == 1 && kW == 1 && kT == 1
    && dW == 1 && dH == 1 && dT == 1
    && padH == 0 && padW == 0 && padT == 0) {
    true
  } else {
    false
  }

  {
    val stdv = 1.0 / math.sqrt(kT * kW * kH * nInputPlane)
    val wInit = RandomUniform(-stdv, stdv)
    val bInit = RandomUniform(-stdv, stdv)

    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.OUT_IN_KT_KH_KW)
    Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    zeroGradParameters()
  }

  private def calculateAdj(targetSize : Int, ker : Int, pad : Int, stride : Int) : Int = {
    (targetSize + 2 * pad - ker) % stride
  }

  private def shapeCheck(input : Tensor[T], gradOutput : Tensor[T],
    weight : Tensor[T], bias : Tensor[T],
    kT: Int, kH : Int, kW : Int,
    dT: Int, dH : Int, dW : Int,
    padT: Int, padH : Int, padW : Int,
    adjT: Int, adjH : Int, adjW : Int) : Unit = {

    require(kT > 0 && kW > 0 && kH > 0,
      s"VolumetricFullConvolution: kernel size should be greater than zero, " +
      s"but got kT: $kT kH: $kH kW: $kW")
    require(dW > 0 && dW > 0 && dH > 0,
      s"VolumetricFullConvolution: stride should be greater than zero, " +
      s"but got dT: $dT dH: $dH dW: $dW")
    require(weight.nDimension == 4 || weight.nDimension == 6,
      s"VolumetricFullConvolution: 4D or 6D weight tensor expected, but got size: ${weight.dim()}")

    if (null != bias) {
      require(bias.nDimension() == 1,
        s"VolumetricFullConvolution: bias should be 1 dim, but got dim:${bias.nDimension()}")
      require(bias.size(1) == weight.size(3) * weight.size(1),
        s"VolumetricFullConvolution: bias's size equals to weight.size(3) * weight.size(1) " +
          s"= ${weight.size(1) * weight.size(3)}, but got size:${bias.size(1)}")
    }

    val ndim = input.nDimension()

    require(ndim == 4 || ndim == 5, s"VolumetricFullConvolution: 4D or 5D input tensor expected, " +
      s"but got size: ${input.dim()}")

    val dimFilter = if (input.dim() == 4) 1 else 2
    val dimDepth = if (input.dim() == 4) 2 else 3
    val dimHeight = if (input.dim() == 4) 3 else 4
    val dimWidth = if (input.dim() == 4) 4 else 5

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)
    val inputDepth = input.size(dimDepth)

    val outputDepth = (inputDepth - 1) * dT - 2 * padT + kT + adjT
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW

    require(outputWidth >= 1 && outputDepth >= 1 && outputHeight >= 1,
      s"VolumetricFullConvolution: Given input size: (${ input.size().mkString("x") })." +
        s" Calculated output size:" +
        s" (${ nOutputPlane }x${ outputDepth }x${ outputHeight }x${ outputWidth })." +
        s" Output size is too small")

    require(input.nDimension() == ndim && input.size(dimFilter) == nInputPlane,
      s"VolumetricFullConvolution: input's feature maps should be $nInputPlane, " +
        s"but got ${input.size(dimFilter)}")

    if (null != gradOutput) {
      require(gradOutput.nDimension() == ndim, s"VolumetricFullConvolution: gradOutput should be " +
        s"$ndim, but got ${gradOutput.nDimension()}")
      require(gradOutput.size(dimFilter) == nOutputPlane
        && gradOutput.size(dimDepth) == outputDepth
        && gradOutput.size(dimHeight) == outputHeight
        && gradOutput.size(dimWidth) == outputWidth,
        s"VolumetricFullConvolution: GradOutput's size should be" +
          s" ($nOutputPlane x $outputDepth x $outputHeight " +
          s"x $outputWidth), but got (${gradOutput.size(dimFilter)} x" +
          s" ${gradOutput.size(dimDepth)} x" +
          s" ${gradOutput.size(dimHeight)} x" +
          s" ${gradOutput.size(dimWidth)})")
    }
  }

  protected def updateOutputFrame(
    input: Tensor[T],
    output: Tensor[T],
    weight: Tensor[T],
    bias: Tensor[T],
    columns: Tensor[T],
    kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int,
    padT: Int, padW: Int, padH: Int,
    nInputPlane: Int,
    inputDepth: Int, inputWidth: Int, inputHeight: Int,
    nOutputPlane: Int,
    outputDepth: Int, outputWidth: Int, outputHeight: Int)
    (implicit ev: TensorNumeric[T]): Unit = {
    val output2d = output.view(nOutputPlane,
      outputDepth * outputHeight * outputWidth)

    // M,N,K are dims of matrix A and B
    // (see https://software.intel.com/en-us/node/468480)
    val m = weight.size(2)
    val n = columns.size(2)
    val k = weight.size(1)

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    DenseTensorBLAS.gemm[T](
      'N', 'T',
      n, m, k,
      ev.one,
      input.storage().array(), input.storageOffset() - 1, n,
      weight.storage().array(), weight.storageOffset() - 1, m,
      ev.zero,
      columns.storage().array(), columns.storageOffset() - 1, n
    )

    if (!_1x1x1) {
      ev.getType() match {
        case DoubleType => NNPrimitive.col2volDouble(
          columns.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputDepth, outputHeight, outputWidth,
          kT, kH, kW,
          padT, padH, padW,
          dT, dH, dW,
          1, 1, 1,
          output2d.asInstanceOf[Tensor[Double]]
        )

        case FloatType => NNPrimitive.col2volFloat(
          columns.asInstanceOf[Tensor[Float]],
          nOutputPlane, outputDepth, outputHeight, outputWidth,
          kT, kH, kW,
          padT, padH, padW,
          dT, dH, dW,
          1, 1, 1,
          output2d.asInstanceOf[Tensor[Float]]
        )

        case _ => throw new UnsupportedOperationException(
          "VolumetricFullConvolution: only Float/Double type supported")
      }
    }

    if (null != bias) {
      output2d.addr(ev.one, bias, onesBias)
    }
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val inputTensor: Tensor[T] = if (input.isInstanceOf[Table]) {
      if (gradInput == null || !gradInput.isInstanceOf[Table]) {
        gradInput = T()
      }
      val targetTensor: Tensor[T] = input.toTable[Tensor[T]](2)
      val tDims = targetTensor.dim()
      val tT = targetTensor.size(tDims - 2)
      val tH = targetTensor.size(tDims - 1)
      val tW = targetTensor.size(tDims)
      adjT = calculateAdj(tT, kT, padT, dT)
      adjW = calculateAdj(tW, kW, padW, dW)
      adjH = calculateAdj(tH, kH, padH, dH)
      input.toTable[Tensor[T]](1)
    } else {
      if (gradInput == null || gradInput.isInstanceOf[Table]) {
        gradInput = Tensor[T]()
      }
      input.toTensor[T]
    }

    shapeCheck(inputTensor, null, weight, bias, kT, kH, kW,
      dT, dH, dW, padT, padH, padW, adjT, adjH, adjW)
    require(inputTensor.isContiguous(), "VolumetricFullConvolution: input should be contiguous")

    val isBatch = if (inputTensor.nDimension() == 4) {
      // Force batch
      inputTensor.resize(1,
        inputTensor.size(1), inputTensor.size(2), inputTensor.size(3), inputTensor.size(4))
      false
    } else {
      true
    }

    val inputWidth = inputTensor.size(5)
    val inputHeight = inputTensor.size(4)
    val inputDepth = inputTensor.size(3)

    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputDepth = (inputDepth - 1) * dT - 2 * padT + kT + adjT

    // Batch size + input planes
    val batchSize = inputTensor.size(1)

    // Resize output
    output.resize(batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth)
    output.zero()

    if (null != bias &&
      (onesBias.dim() != 1 || onesBias.size(1) != outputDepth * outputHeight * outputWidth)) {
      onesBias.resize(Array(outputDepth * outputHeight * outputWidth)).fill(ev.one)
    }

    if (_1x1x1) {
      columns.set(inputTensor)
      columns.resize(Array(batchSize, nGroup, kT * kW * kH * nOutputPlane / nGroup,
        inputDepth * inputHeight * inputWidth))
    } else {
      columns.resize(Array(batchSize, nGroup, kT * kW * kH * nOutputPlane / nGroup,
        inputDepth * inputHeight * inputWidth))
    }
    columns.zero()

    if (weightMM == null) {
      weightMM = weight.view(nGroup, nInputPlane / nGroup,
        nOutputPlane * kT * kH * kW / nGroup)
    }

    // Define a buffer of ones, for bias accumulation
    // Note: this buffer can be shared with other modules, it only ever gets increased,
    // and always contains ones.
    if (ones.nDimension != 3 || ones.size(1) * ones.size(2)
      * ones.size(3) < outputDepth * outputHeight * outputWidth)
    {
      // Resize plane and fill with ones...
      ones.resize(outputDepth, outputHeight, outputWidth)
      ones.fill(ev.one)
    }

    var elt = 1
    // For each element in batch, do:
    while(elt <= batchSize) {
      // Matrix mulitply per output:
      val input_n = inputTensor.select(1, elt)
      require(input_n.isContiguous(),
        s"VolumetricFullConvolution: input($elt) should be contiguous")
      val output_n = output.select(1, elt)
      val columns_n = columns.select(1, elt)

      var g = 0
      while (g < nGroup) {
        val bias_g = if (!noBias) {
          bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup)
        } else {
          null
        }
        updateOutputFrame(
          input_n.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          output_n.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1),
          bias_g,
          columns_n.select(1, g + 1),
          kT, kW, kH, dT, dW, dH,
          padT, padW, padH,
          nInputPlane / nGroup, inputDepth, inputWidth, inputHeight,
          nOutputPlane / nGroup, outputDepth, outputWidth, outputHeight)
        g += 1
      }
      elt += 1
    }

    // Resize output
    if(!isBatch) {
      output.resize(nOutputPlane, outputDepth, outputHeight, outputWidth)
      inputTensor.resize(nInputPlane, inputDepth, inputHeight, inputWidth)
    }

    output
  }

  protected def updateGradInputFrame(
    gradInput: Tensor[T], gradOutput: Tensor[T],
    weight: Tensor[T], columns: Tensor[T],
    kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int,
    padT: Int, padW: Int, padH: Int,
    outputDepth: Int, outputHeight: Int, outputWidth: Int)(implicit ev: TensorNumeric[T]): Unit = {
    // Extract columns:
    ev.getType() match {
      case DoubleType => NNPrimitive.vol2colDouble(
        gradOutput.asInstanceOf[Tensor[Double]],
        gradOutput.size(1), outputDepth, outputHeight, outputWidth,
        kT, kH, kW,
        padT, padH, padW,
        dT, dH, dW,
        1, 1, 1,
        columns.asInstanceOf[Tensor[Double]]
      )

      case FloatType => NNPrimitive.vol2colFloat(
        gradOutput.asInstanceOf[Tensor[Float]],
        gradOutput.size(1), outputDepth, outputHeight, outputWidth,
        kT, kH, kW,
        padT, padH, padW,
        dT, dH, dW,
        1, 1, 1,
        columns.asInstanceOf[Tensor[Float]]
      )

      case _ => throw new UnsupportedOperationException(
        s"VolumetricFullConvolution: only Float/Double type supported")
    }

    // M,N,K are dims of matrix A and B
    // (see https://software.intel.com/en-us/node/468480)
    val m = weight.size(1)
    val n = columns.size(2)
    val k = weight.size(2)

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    DenseTensorBLAS.gemm[T](
      'N', 'N',
      n, m, k,
      ev.one,
      columns.storage().array(), columns.storageOffset() - 1, n,
      weight.storage().array(), weight.storageOffset() - 1, k,
      ev.zero,
      gradInput.storage().array(), gradInput.storageOffset() - 1, n
    )

  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    val inputTensor: Tensor[T] = if (input.isInstanceOf[Table]) {
      input.toTable[Tensor[T]](1)
    } else {
      input.toTensor[T]
    }
    val gradInputTensor: Tensor[T] = if (input.isInstanceOf[Table]) {
      if (!gradInput.toTable.contains(1)) {
        gradInput.toTable(1) = Tensor[T]()
      }
      gradInput.toTable[Tensor[T]](1)
    } else {
      gradInput.toTensor[T]
    }
    shapeCheck(
      inputTensor,
      gradOutput,
      weight,
      null,
      kT, kH, kW,
      dT, dH, dW,
      padT, padH, padW,
      adjT, adjH, adjW)

    val isBatch = if (inputTensor.nDimension() == 4) {
      // Force batch
      inputTensor.resize(1,
        inputTensor.size(1), inputTensor.size(2), inputTensor.size(3), inputTensor.size(4))
      gradOutput.resize(1,
        gradOutput.size(1), gradOutput.size(2), gradOutput.size(3), gradOutput.size(4))
      false
    } else {
      true
    }

    val inputWidth = inputTensor.size(5)
    val inputHeight = inputTensor.size(4)
    val inputDepth = inputTensor.size(3)
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputDepth = (inputDepth - 1) * dT - 2 * padT + kT + adjT

    // Batch size + input planes
    val batchSize = inputTensor.size(1)

    gradInputTensor.resizeAs(inputTensor)
    gradInputTensor.zero()

    if (_1x1x1) {
      columns.set(gradInputTensor)
      columns.resize(Array(batchSize, nGroup, kT * kW * kH * nOutputPlane / nGroup,
        inputDepth * inputHeight * inputWidth))
    } else {
      columns.resize(Array(batchSize, nGroup, kT * kW * kH * nOutputPlane / nGroup,
        inputDepth * inputHeight * inputWidth))
    }

    var elt = 1
    // For each element in batch, do:
    while (elt <= batchSize) {
      // Matrix mulitply per sample:
      val gradInput_n = gradInputTensor.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)
      val columns_n = columns.select(1, elt)

      var g = 0
      while (g < nGroup) {
        updateGradInputFrame(
          gradInput_n.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          gradOutput_n.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1),
          columns_n.select(1, g + 1),
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH,
          outputDepth, outputHeight, outputWidth)
        g += 1
      }

      elt += 1
    }

    // Resize output
    if (!isBatch) {
      gradOutput.resize(nOutputPlane, outputDepth, outputHeight, outputWidth)
      inputTensor.resize(nInputPlane, inputDepth, inputHeight, inputWidth)
      gradInputTensor.resize(nInputPlane, inputDepth, inputHeight, inputWidth)
    }

    if (input.isInstanceOf[Table]) {
      val input2 = input.toTable[Tensor[T]](2)
      zeroScalar.resizeAs(input2).zero()
      ones.resizeAs(input2).fill(ev.one)
      val zeroTensor = zeroScalar.view(ones.size()).expandAs(input2)
      gradInput.toTable(1) = gradInputTensor
      gradInput.toTable(2) = zeroTensor
    }

    gradInput
  }

  protected def calcGradParametersFrame(
    input: Tensor[T], gradOutput: Tensor[T], gradWeight: Tensor[T],
    gradBias: Tensor[T], columns: Tensor[T],
    outputDepth: Int, outputHeight: Int, outputWidth: Int,
    scaleW: T, scaleB: T)(implicit ev: TensorNumeric[T]): Unit = {
    // Extract columns:
    ev.getType() match {
      case DoubleType => NNPrimitive.vol2colDouble(
        gradOutput.asInstanceOf[Tensor[Double]],
        gradOutput.size(1), outputDepth, outputHeight, outputWidth,
        kT, kH, kW,
        padT, padH, padW,
        dT, dH, dW,
        1, 1, 1,
        columns.asInstanceOf[Tensor[Double]]
      )

      case FloatType => NNPrimitive.vol2colFloat(
        gradOutput.asInstanceOf[Tensor[Float]],
        gradOutput.size(1), outputDepth, outputHeight, outputWidth,
        kT, kH, kW,
        padT, padH, padW,
        dT, dH, dW,
        1, 1, 1,
        columns.asInstanceOf[Tensor[Float]]
      )
      case t => throw new NotImplementedError(s"$t is not supported")
    }

    // M,N,K are dims of matrix A and B
    // (see https://software.intel.com/en-us/node/468480)
    val n = columns.size(1)   // nOutputPlane * kt * kh * kw
    var m = input.size(1)   // nInputPlane
    var k = columns.size(2)   // inputDepth * inputHeight * inputWidth

    if (scaleW != 0) {
      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        'T', 'N',
        n, m, k,
        scaleW,
        columns.storage().array(), columns.storageOffset() - 1, k,
        input.storage().array(), input.storageOffset() - 1, k,
        ev.one,
        gradWeight.storage().array(), gradWeight.storageOffset() - 1, n
      )
    }
    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see https://software.intel.com/en-us/node/468480)
    m = nOutputPlane
    k = outputDepth * outputHeight * outputWidth
    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (null != gradBias && scaleB != 0) {
      ev.gemv(
        'T',
        k, m,
        scaleB,
        gradOutput.storage().array(), gradOutput.storageOffset() - 1, k,
        ones.storage().array(), ones.storageOffset() - 1, 1,
        ev.one,
        gradBias.storage().array(), gradBias.storageOffset() - 1, 1
      )
    }
  }


  override def accGradParameters(input: Activity, gradOutput: Tensor[T]): Unit = {
    val inputTensor: Tensor[T] = if (input.isInstanceOf[Table]) {
      val targetTensor: Tensor[T] = input.toTable[Tensor[T]](2)
      val tDims = targetTensor.dim()
      val tT = targetTensor.size(tDims - 2)
      val tH = targetTensor.size(tDims - 1)
      val tW = targetTensor.size(tDims)
      adjT = calculateAdj(tT, kT, padT, dT)
      adjW = calculateAdj(tW, kW, padW, dW)
      adjH = calculateAdj(tH, kH, padH, dH)
      input.toTable[Tensor[T]](1)
    } else {
      input.toTensor
    }

    shapeCheck(inputTensor, gradOutput, gradWeight, gradBias,
      kT, kH, kW,
      dT, dH, dW,
      padT, padH, padW,
      adjT, adjH, adjW)

    val isBatch = if (inputTensor.nDimension() == 4) {
      // Force batch
      inputTensor.resize(1,
        inputTensor.size(1), inputTensor.size(2), inputTensor.size(3), inputTensor.size(4))
      gradOutput.resize(1,
        gradOutput.size(1), gradOutput.size(2), gradOutput.size(3), gradOutput.size(4))
      false
    } else {
      true
    }

    val inputDepth = inputTensor.size(3)
    val inputHeight = inputTensor.size(4)
    val inputWidth = inputTensor.size(5)
    val outputDepth = (inputDepth - 1) * dT - 2 * padT + kT + adjT
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH

    // Batch size + input planes
    val batchSize = inputTensor.size(1)

    gradWeightMMInBatch.resize(Array(batchSize, nGroup, nInputPlane / nGroup,
      nOutputPlane * kT * kH * kW / nGroup))
    gradWeightMMInBatch.zero()
    if (!noBias) {
      gradientBiasMT.resize(Array(batchSize, nOutputPlane))
      gradientBiasMT.zero()
    }

    // Define a buffer of ones, for bias accumulation
    if (ones.nDimension != 2 || ones.size(1) * ones.size(2) < outputHeight * outputWidth) {
      // Resize plane and fill with ones...
      ones.resize(outputHeight, outputWidth)
      ones.fill(ev.one)
    }

    if (onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
      onesBatch.resize(Array(batchSize)).fill(ev.one)
    }

    var elt = 1
    // For each element in batch, do:
    while (elt <= batchSize) {
      // Matrix mulitply per output:
      val input_n = inputTensor.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)
      val column_n = columns.select(1, elt)
      var g = 0
      while (g < nGroup) {
        val gradBias_G = if (noBias) {
          null
        } else if (isBatch) {
          gradientBiasMT.select(1, elt).narrow(1, g * nOutputPlane / nGroup + 1,
            nOutputPlane / nGroup)
        } else {
          gradBias.narrow(1, g * nOutputPlane / nGroup + 1,
            nOutputPlane / nGroup)
        }
        calcGradParametersFrame(
          input_n.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          gradOutput_n.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          gradWeightMMInBatch.select(1, elt).select(1, g + 1),
          gradBias_G,
          column_n.select(1, g + 1),
          outputDepth, outputHeight, outputWidth,
          ev.fromType[Double](scaleW),
          ev.fromType[Double](scaleB))
        g += 1
      }

      elt += 1
    }

    val gradView = gradWeightMMInBatch.view(batchSize,
      nOutputPlane * nInputPlane * kT * kH * kW / nGroup).t()
    val grad = gradWeight.view(nOutputPlane * nInputPlane * kT * kH * kW / nGroup)
    grad.addmv(ev.one, ev.one, gradView, onesBatch)
    if (!noBias) gradBias.addmv(ev.one, ev.one, gradientBiasMT.t(), onesBatch)

    // Resize
    if (!isBatch) {
      gradOutput.resize(nOutputPlane, outputDepth, outputHeight, outputWidth)
      inputTensor.resize(nInputPlane, inputDepth, inputHeight, inputWidth)
    }

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }

  override def clearState() : this.type = {
    super.clearState()
    columns.set()
    ones.set()
    zeroScalar.set()
    if (onesBias != null) onesBias.set()
    onesBatch.set()
    weightMM = null
    if (gradientBiasMT != null) gradientBiasMT.set()
    gradWeightMMInBatch.set()
    this
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[VolumetricFullConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[VolumetricFullConvolution[T]]
    if (this.eq(other)) {
      return true
    }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      adjW == other.adjW &&
      adjH == other.adjH &&
      weight == other.weight &&
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kT.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dT.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padT.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + adjT.hashCode()
    hash = hash * seed + adjW.hashCode()
    hash = hash * seed + adjH.hashCode()
    hash = hash * seed + weight.hashCode()
    if (!noBias) hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    if (!noBias) hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName()}($nInputPlane -> $nOutputPlane, " +
      s"$kT x $kW x $kH, $dT x $dW, $dH, " +
      s"$padT, $padW, $padH, $adjT, $adjW, $adjH)"
  }
}

object VolumetricFullConvolution {
  def apply[@specialized(Float, Double) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kT: Int,
    kW: Int,
    kH: Int,
    dT: Int = 1,
    dW: Int = 1,
    dH: Int = 1,
    padT: Int = 0,
    padW: Int = 0,
    padH: Int = 0,
    adjT: Int = 0,
    adjW: Int = 0,
    adjH: Int = 0,
    nGroup: Int = 1,
    noBias: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]) : VolumetricFullConvolution[T] = {
    new VolumetricFullConvolution[T](nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH,
      padT, padW, padH, adjT, adjW, adjH, nGroup, noBias,
      wRegularizer, bRegularizer)
  }
}

