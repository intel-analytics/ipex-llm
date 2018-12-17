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
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{Shape, T, Table}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * Apply a 2D full convolution over an input image.
 *
 * The input tensor is expected to be a 3D or 4D(with batch) tensor. Note that instead
 * of setting adjW and adjH, SpatialFullConvolution[Table, T] also accepts a table input
 * with two tensors: T(convInput, sizeTensor) where convInput is the standard input tensor,
 * and the size of sizeTensor is used to set the size of the output (will ignore the adjW and
 * adjH values used to construct the module). This module can be used without a bias by setting
 * parameter noBias = true while constructing the module.
 *
 * If input is a 3D tensor nInputPlane x height x width,
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
 * @param kW The kernel width of the convolution.
 * @param kH The kernel height of the convolution.
 * @param dW The step of the convolution in the width dimension. Default is 1.
 * @param dH The step of the convolution in the height dimension. Default is 1.
 * @param padW The additional zeros added per width to the input planes. Default is 0.
 * @param padH The additional zeros added per height to the input planes. Default is 0.
 * @param adjW Extra width to add to the output image. Default is 0.
 * @param adjH Extra height to add to the output image. Default is 0.
 * @param nGroup Kernel group number.
 * @param noBias If bias is needed.
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */

@SerialVersionUID(- 3110412775551642284L)
class SpatialFullConvolution[T: ClassTag](
  val nInputPlane: Int,
  val nOutputPlane: Int,
  val kW: Int,
  val kH: Int,
  val dW: Int = 1,
  val dH: Int = 1,
  val padW: Int = 0,
  val padH: Int = 0,
  var adjW: Int = 0,
  var adjH: Int = 0,
  val nGroup: Int = 1,
  val noBias: Boolean = false,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T])
  extends AbstractModule[Activity, Tensor[T], T] with Initializable {

  require(adjW <= dW - 1 && adjH <= dH - 1,
    "SpatialFullConvolution: adjW=$adjW and adjH=$adjH must be smaller than " +
      s"(dW - 1)=${dW - 1} and (dH - 1)=${dH - 1} respectively")

  val weight: Tensor[T] = Tensor[T](nGroup, nInputPlane / nGroup,
    nOutputPlane / nGroup, kH, kW)
  val bias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)

  val gradWeight: Tensor[T] = Tensor[T](nGroup, nInputPlane / nGroup, nOutputPlane / nGroup, kH, kW)
  val gradBias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)
  private val columns: Tensor[T] = Tensor[T]()
  private val ones: Tensor[T] = Tensor[T]()
  private val zeroScalar: Tensor[T] = Tensor[T]()
  protected val onesBias = Tensor[T]()
  protected val onesBatch = Tensor[T]()
  protected var weightMM: Tensor[T] = null
  protected val gradientBiasMT: Tensor[T] = Tensor[T]()
  protected val gradWeightMMInBatch: Tensor[T] = Tensor[T]()

  protected val _1x1 = if (kH == 1 && kW == 1 && dW == 1 && dH == 1
    && padH == 0 && padW == 0) {
    true
  } else {
    false
  }

  {
    val stdv = 1.0 / math.sqrt(kW * kH * nInputPlane)
    val wInit = RandomUniform(-stdv, stdv)
    val bInit = RandomUniform(-stdv, stdv)

    setInitMethod(wInit, bInit)
  }

  private var im2colTime = 0L
  private var col2imTime = 0L

  def getIm2ColTime(): Double = im2colTime

  def getCol2ImgTime(): Double = col2imTime

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.GP_IN_OUT_KW_KH)
    Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    zeroGradParameters()
  }

  private def calculateAdj(targetSize : Int, ker : Int, pad : Int, stride : Int) : Int = {
    (targetSize + 2 * pad - ker) % stride
  }

  private def shapeCheck(input : Tensor[T], gradOutput : Tensor[T],
    weight : Tensor[T], bias : Tensor[T],
    kH : Int, kW : Int,
    dH : Int, dW : Int,
    padH : Int, padW : Int,
    adjH : Int, adjW : Int) : Unit = {

    require(kW > 0 && kH > 0, s"SpatialFullConvolution: kernel size should be greater than zero, " +
      s"but got kH: $kH kW: $kW")
    require(dW > 0 && dH > 0, s"SpatialFullConvolution: stride should be greater than zero, " +
      s"but got dH: $dH dW: $dW")
    require(weight.nDimension == 3 || weight.nDimension == 5,
      s"SpatialFullConvolution: 3D or 5D weight tensor expected, but got size: ${weight.dim()}")

    if (null != bias) {
      require(bias.nDimension() == 1,
        s"SpatialFullConvolution: bias should be 1 dim, but got dim:${bias.nDimension()}")
      require(bias.size(1) == weight.size(3) * weight.size(1),
        s"SpatialFullConvolution: bias's size equals to weight.size(3) * weight.size(1) " +
          s"= ${weight.size(1) * weight.size(3)}, but got size:${bias.size(1)}")
    }

    val ndim = input.nDimension
    val dimf = if (ndim == 4) 2 else 1
    val dimh = if (ndim == 4) 3 else 2
    val dimw = if (ndim == 4) 4 else 3

    require(ndim == 3 || ndim == 4, s"SpatialFullConvolution: 3D or 4D input tensor expected, " +
      s"but got size: ${input.dim()}")

    val inputHeight = input.size(dimh)
    val inputWidth = input.size(dimw)
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW

    require(outputWidth >= 1 || outputHeight >= 1,
      s"SpatialFullConvolution: Given input size: ($nInputPlane x $inputHeight x $inputWidth). " +
      s"Calculated output size: ($nOutputPlane x $outputHeight x $outputWidth). " +
      s"Output size is too small")

    require(input.nDimension() == ndim && input.size(dimf) == nInputPlane,
      s"SpatialFullConvolution: input's feature maps should be $nInputPlane, " +
        s"but got ${input.size(dimf)}")

    if (null != gradOutput) {
      require(gradOutput.nDimension() == ndim, s"SpatialFullConvolution: gradOutput should be " +
        s"$ndim, but got ${gradOutput.nDimension()}")
      require(gradOutput.size(dimf) == nOutputPlane
        && gradOutput.size(dimh) == outputHeight
        && gradOutput.size(dimw) == outputWidth,
        s"SpatialFullConvolution: GradOutput's size should be (${nOutputPlane} x ${outputHeight} " +
          s"x ${outputWidth}), but got (${gradOutput.size(dimf)} x ${gradOutput.size(dimh)} " +
          s"x ${gradOutput.size(dimw)})")
    }
  }

  protected def updateOutputFrame(
      input: Tensor[T], output: Tensor[T], weight: Tensor[T],
      bias: Tensor[T], columns: Tensor[T],
      kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
      nInputPlane: Int,
      inputWidth: Int, inputHeight: Int,
      nOutputPlane: Int,
      outputWidth: Int, outputHeight: Int)(implicit ev: TensorNumeric[T]): Unit = {
    val output2d = output.view(nOutputPlane, outputHeight * outputWidth)

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

    if (!_1x1) {
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.col2imWithDilationDouble(
          columns.asInstanceOf[Tensor[Double]], output2d.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case FloatType => NNPrimitive.col2imWithDilationFloat(
          columns.asInstanceOf[Tensor[Float]], output2d.asInstanceOf[Tensor[Float]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case _ => throw new UnsupportedOperationException(
          "SpatialFullConvolution: only Float/Double type supported")
      }
      col2imTime += System.nanoTime() - before
    }
    if (null != bias) {
      output2d.addr(ev.one, bias, onesBias)
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"Deconvolution2D requires 4D input, but got input dim ${input.length}")
    val inputHeight = input(2)
    val inputWidth = input(3)
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    Shape(input(0), nOutputPlane, outputHeight, outputWidth)
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val inputTensor: Tensor[T] = if (input.isInstanceOf[Table]) {
      if (gradInput == null || !gradInput.isInstanceOf[Table]) {
        gradInput = T()
      }
      val targetTensor: Tensor[T] = input.toTable[Tensor[T]](2)
      val tDims = targetTensor.dim()
      val tH = targetTensor.size(tDims - 1)
      val tW = targetTensor.size(tDims)
      adjW = calculateAdj(tW, kW, padW, dW)
      adjH = calculateAdj(tH, kH, padH, dH)
      input.toTable[Tensor[T]](1)
    } else {
      if (gradInput == null || gradInput.isInstanceOf[Table]) {
        gradInput = Tensor[T]()
      }
      input.toTensor[T]
    }

    shapeCheck(inputTensor, null, weight, bias, kH, kW, dH, dW, padH, padW, adjH, adjW)
    require(inputTensor.isContiguous(), "SpatialFullConvolution: input should be contiguous")

    val isBatch = if (inputTensor.nDimension() == 3) {
      // Force batch
      inputTensor.resize(1, inputTensor.size(1), inputTensor.size(2), inputTensor.size(3))
      false
    } else {
      true
    }

    val inputHeight = inputTensor.size(3)
    val inputWidth = inputTensor.size(4)

    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW

    // Batch size + input planes
    val batchSize = inputTensor.size(1)

    // Resize output
    output.resize(batchSize, nOutputPlane, outputHeight, outputWidth)
    output.zero()

    if (onesBias.dim() != 1 || onesBias.size(1) != outputHeight * outputWidth) {
      onesBias.resize(Array(outputHeight * outputWidth)).fill(ev.one)
    }

    if (_1x1) {
      columns.set(inputTensor)
      columns.resize(Array(batchSize, nGroup, kW * kH * nOutputPlane / nGroup,
        inputHeight * inputWidth))
    } else {
      columns.resize(Array(batchSize, nGroup, kW * kH * nOutputPlane / nGroup,
        inputHeight * inputWidth))
    }

    // weight's storage might change, so make a view every time
    weightMM = weight.view(nGroup, nInputPlane / nGroup,
        nOutputPlane * kH * kW / nGroup)

    var elt = 1
    // For each element in batch, do:
    while(elt <= batchSize) {
      // Matrix mulitply per output:
      val input_n = inputTensor.select(1, elt)
      require(input_n.isContiguous(), s"SpatialFullConvolution: input($elt) should be contiguous")
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
          kW, kH, dW, dH,
          padW, padH,
          nInputPlane / nGroup, inputWidth, inputHeight,
          nOutputPlane / nGroup, outputWidth, outputHeight)
        g += 1
      }
      elt += 1
    }

    // Resize output
    if(!isBatch) {
      output.resize(nOutputPlane, outputHeight, outputWidth)
      inputTensor.resize(nInputPlane, inputHeight, inputWidth)
    }

    output
  }

  protected def updateGradInputFrame(
      gradInput: Tensor[T], gradOutput: Tensor[T],
      weight: Tensor[T], columns: Tensor[T],
      kW: Int, kH: Int,
      dW: Int, dH: Int,
      padW: Int, padH: Int,
      outputHeight: Int, outputWidth: Int)(implicit ev: TensorNumeric[T]): Unit = {
    // Extract columns:
    val before = System.nanoTime()
    ev.getType() match {
      case DoubleType => NNPrimitive.im2colWithDilationDouble(
        gradOutput.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
        gradOutput.size(1), outputHeight, outputWidth,
        kH, kW,
        padH, padW,
        dH, dW,
        1, 1
      )

      case FloatType => NNPrimitive.im2colWithDilationFloat(
        gradOutput.asInstanceOf[Tensor[Float]], columns.asInstanceOf[Tensor[Float]],
        gradOutput.size(1), outputHeight, outputWidth,
        kH, kW,
        padH, padW,
        dH, dW,
        1, 1
      )

      case _ => throw new UnsupportedOperationException(
        s"SpatialFullConvolution: only Float/Double type supported")
    }
    im2colTime += System.nanoTime() - before

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
    shapeCheck(inputTensor, gradOutput, weight, null, kH, kW, dH, dW, padH, padW, adjH, adjW)

    val isBatch = if (inputTensor.nDimension() == 3) {
      // Force batch
      inputTensor.resize(1, inputTensor.size(1), inputTensor.size(2), inputTensor.size(3))
      gradOutput.resize(1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3))
      false
    } else {
      true
    }

    val inputWidth = inputTensor.size(4)
    val inputHeight = inputTensor.size(3)
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH

    // Batch size + input planes
    val batchSize = inputTensor.size(1)

    gradInputTensor.resizeAs(inputTensor)
    gradInputTensor.zero()

    if (_1x1) {
      columns.set(gradInputTensor)
      columns.resize(Array(batchSize, nGroup, kW * kH * nOutputPlane / nGroup,
        inputHeight * inputWidth))
    } else {
      columns.resize(Array(batchSize, nGroup, kW * kH * nOutputPlane / nGroup,
        inputHeight * inputWidth))
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
          kW, kH, dW, dH, padW, padH, outputHeight, outputWidth)
        g += 1
      }

      elt += 1
    }

    // Resize output
    if (!isBatch) {
      gradOutput.resize(nOutputPlane, outputHeight, outputWidth)
      inputTensor.resize(nInputPlane, inputHeight, inputWidth)
      gradInputTensor.resize(nInputPlane, inputHeight, inputWidth)
    }

    if (input.isInstanceOf[Table]) {
      val input2 = input.toTable[Tensor[T]](2)
      zeroScalar.resizeAs(input2).zero()
      ones.resizeAs(input2).fill(ev.one)
      val zeroTensor = zeroScalar.view(ones.size()).expandAs(input2)
      gradInput.toTable(1) = gradInputTensor
      gradInput.toTable(2) = zeroTensor
    }

    return gradInput
  }

  protected def calcGradParametersFrame(
      input: Tensor[T], gradOutput: Tensor[T], gradWeight: Tensor[T],
      gradBias: Tensor[T], columns: Tensor[T],
      outputHeight: Int, outputWidth: Int,
      scaleW: T, scaleB: T)(implicit ev: TensorNumeric[T]): Unit = {
    // Extract columns:
    val before = System.nanoTime()
    ev.getType() match {
      case DoubleType => NNPrimitive.im2colWithDilationDouble(
        gradOutput.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
        gradOutput.size(1), outputHeight, outputWidth,
        kH, kW,
        padH, padW,
        dH, dW,
        1, 1
      )

      case FloatType => NNPrimitive.im2colWithDilationFloat(
        gradOutput.asInstanceOf[Tensor[Float]], columns.asInstanceOf[Tensor[Float]],
        gradOutput.size(1), outputHeight, outputWidth,
        kH, kW,
        padH, padW,
        dH, dW,
        1, 1
      )
      case t => throw new NotImplementedError(s"$t is not supported")
    }
    im2colTime += System.nanoTime() - before

    // M,N,K are dims of matrix A and B
    // (see https://software.intel.com/en-us/node/468480)
    val n = columns.size(1)   // nOutputPlane * kh * kw
    var m = input.size(1)   // nInputPlane
    var k = columns.size(2)   // inputHeight * inputWidth

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
    m = gradOutput.size(1)
    k = outputHeight * outputWidth
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
      val tH = targetTensor.size(tDims - 1)
      val tW = targetTensor.size(tDims)
      adjW = calculateAdj(tW, kW, padW, dW)
      adjH = calculateAdj(tH, kH, padH, dH)
      input.toTable[Tensor[T]](1)
    } else {
      input.toTensor
    }

    shapeCheck(inputTensor, gradOutput, gradWeight, gradBias,
      kH, kW, dH, dW, padH, padW, adjH, adjW)

    val isBatch = if (inputTensor.nDimension() == 3) {
      // Force batch
      inputTensor.resize(1, inputTensor.size(1), inputTensor.size(2), inputTensor.size(3))
      gradOutput.resize(1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3))
      false
    } else {
      true
    }

    val inputWidth = inputTensor.size(4)
    val inputHeight = inputTensor.size(3)
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH

    // Batch size + input planes
    val batchSize = inputTensor.size(1)

    gradWeightMMInBatch.resize(Array(batchSize, nGroup, nInputPlane / nGroup,
      nOutputPlane * kH * kW / nGroup))
    gradWeightMMInBatch.zero()
    gradientBiasMT.resize(Array(batchSize, nOutputPlane))

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
          outputHeight, outputWidth,
          ev.fromType[Double](scaleW),
          ev.fromType[Double](scaleB))
        g += 1
      }

      elt += 1
    }

    val gradView = gradWeightMMInBatch.view(batchSize,
      nOutputPlane * nInputPlane * kH * kW / nGroup).t
    val grad = gradWeight.view(nOutputPlane * nInputPlane * kH * kW / nGroup)
    grad.addmv(ev.one, ev.one, gradView, onesBatch)
    if (!noBias) gradBias.addmv(ev.one, ev.one, gradientBiasMT.t, onesBatch)

    // Resize
    if (!isBatch) {
      gradOutput.resize(nOutputPlane, outputHeight, outputWidth)
      inputTensor.resize(nInputPlane, inputHeight, inputWidth)
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
    onesBias.set()
    onesBatch.set()
    weightMM = null
    gradientBiasMT.set()
    gradWeightMMInBatch.set()
    im2colTime = 0L
    col2imTime = 0L
    this
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialFullConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialFullConvolution[T]]
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
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + adjW.hashCode()
    hash = hash * seed + adjH.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane -> $nOutputPlane, " +
      s"$kW x $kH, $dW, $dH, $padW, $padH, $adjW, $adjH)"
  }
}

object SpatialFullConvolution extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int,
      nOutputPlane: Int,
      kW: Int,
      kH: Int,
      dW: Int = 1,
      dH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      adjW: Int = 0,
      adjH: Int = 0,
      nGroup: Int = 1,
      noBias: Boolean = false,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]) : SpatialFullConvolution[T] = {
    new SpatialFullConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, adjW, adjH, nGroup, noBias,
      wRegularizer, bRegularizer)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap
    val intParams = DataConverter.getAttributeValue(context, attrMap.get("intParams")).
      asInstanceOf[Array[Int]]

    val noBias = DataConverter.getAttributeValue(context, attrMap.get("noBias")).
      asInstanceOf[Boolean]

    val wRegularizer = DataConverter.getAttributeValue(context, attrMap.get("wRegularizer")).
      asInstanceOf[Regularizer[T]]

    val bRegularizer = DataConverter.getAttributeValue(context, attrMap.get("bRegularizer")).
      asInstanceOf[Regularizer[T]]

    val fullConv = SpatialFullConvolution(intParams(0), intParams(1), intParams(2), intParams(3),
      intParams(4), intParams(5), intParams(6), intParams(7), intParams(8), intParams(9),
      intParams(10), noBias, wRegularizer, bRegularizer)
    fullConv
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                            fullConvBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    val fullConv = context.moduleData.module.asInstanceOf[SpatialFullConvolution[T]]

    val intParamsBuilder = AttrValue.newBuilder
    val intParams = Array(fullConv.nInputPlane, fullConv.nOutputPlane, fullConv.kW,
      fullConv.kH, fullConv.dW, fullConv.dH, fullConv.padW, fullConv.padH, fullConv.adjW,
      fullConv.adjH, fullConv.nGroup)
    DataConverter.setAttributeValue(context, intParamsBuilder, intParams,
      universe.typeOf[Array[Int]])
    fullConvBuilder.putAttr("intParams", intParamsBuilder.build)

    val biasBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, biasBuilder,
      fullConv.noBias, universe.typeOf[Boolean])
    fullConvBuilder.putAttr("noBias", biasBuilder.build)

    val wRegularizerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, wRegularizerBuilder,
      fullConv.wRegularizer,
      ModuleSerializer.regularizerType)
    fullConvBuilder.putAttr("wRegularizer", wRegularizerBuilder.build)

    val bRegularizerBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      bRegularizerBuilder, fullConv.bRegularizer,
      ModuleSerializer.regularizerType)
    fullConvBuilder.putAttr("bRegularizer", bRegularizerBuilder.build)

  }
}
