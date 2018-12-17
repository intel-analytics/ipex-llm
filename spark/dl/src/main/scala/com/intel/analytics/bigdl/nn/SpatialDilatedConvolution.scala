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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{DenseTensorBLAS, DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T, Table}

import scala.reflect.ClassTag

/**
 * Apply a 2D dilated convolution over an input image.
 *
 * The input tensor is expected to be a 3D or 4D(with batch) tensor.
 *
 * If input is a 3D tensor nInputPlane x height x width,
 * owidth  = floor(width + 2 * padW - dilationW * (kW-1) - 1) / dW + 1
 * oheight = floor(height + 2 * padH - dilationH * (kH-1) - 1) / dH + 1
 *
 * Reference Paper: Yu F, Koltun V. Multi-scale context aggregation by dilated convolutions[J].
 * arXiv preprint arXiv:1511.07122, 2015.
 *
 * @param nInputPlane The number of expected input planes in the image given into forward().
 * @param nOutputPlane The number of output planes the convolution layer will produce.
 * @param kW The kernel width of the convolution.
 * @param kH The kernel height of the convolution.
 * @param dW The step of the convolution in the width dimension. Default is 1.
 * @param dH The step of the convolution in the height dimension. Default is 1.
 * @param padW The additional zeros added per width to the input planes. Default is 0.
 * @param padH The additional zeros added per height to the input planes. Default is 0.
 * @param dilationW The number of pixels to skip. Default is 1.
 * @param dilationH The number of pixels to skip. Default is 1.
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */

@SerialVersionUID(- 933818099759912492L)
class SpatialDilatedConvolution[T: ClassTag](
  val nInputPlane: Int,
  val nOutputPlane: Int,
  val kW: Int,
  val kH: Int,
  val dW: Int = 1,
  val dH: Int = 1,
  val padW: Int = 0,
  val padH: Int = 0,
  val dilationW: Int = 1,
  val dilationH: Int = 1,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  val weight: Tensor[T] = Tensor[T](nOutputPlane, nInputPlane, kH, kW)
  val gradWeight = Tensor[T](nOutputPlane, nInputPlane, kH, kW)
  val gradBias = Tensor[T](nOutputPlane)

  val bias: Tensor[T] = Tensor[T](nOutputPlane)
  @transient private var fInput: Tensor[T] = null
  @transient private var fGradInput: Tensor[T] = null

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
    weightInitMethod.init(weight, VariableFormat.OUT_IN_KW_KH)
    biasInitMethod.init(bias, VariableFormat.ONE_D)
    zeroGradParameters()
  }

  private def shapeCheck(
    input: Tensor[T], gradOutput: Tensor[T],
    weight: Tensor[T], bias: Tensor[T],
    kH: Int, kW: Int, dH: Int, dW: Int, padH: Int, padW: Int,
    dilationH: Int, dilationW: Int) {

    require(weight.nDimension == 4,
      "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, " +
        s"but got: ${weight.nDimension()}")
    require(kW > 0 && kH > 0,
      s"kernel size should be greater than zero, but got kH: $kH kW: $kW")
    require(dW > 0 && dH > 0,
      s"stride should be greater than zero, but got dH: $dH dW: $dW")
    require(weight.nDimension == 2 || weight.nDimension == 4,
      s"2D or 4D weight tensor expected, but got: ${weight.nDimension()}")

    if (null != bias) {
      require(bias.nDimension() == 1 && bias.size(1) == weight.size(1))
    }

    val nDim = input.nDimension
    val dimF = if (nDim == 4) 2 else 1
    val dimH = if (nDim == 4) 3 else 2
    val dimW = if (nDim == 4) 4 else 3

    require(nDim == 3 || nDim == 4,
      "SpatialDilatedConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)

    val inputHeight = input.size(dimH)
    val inputWidth = input.size(dimW)
    val outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1
    val outputWidth = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1

    require(outputWidth >= 1 || outputHeight >= 1,
      s"Given input size: ($nInputPlane x $inputHeight x $inputWidth)" +
        s"Calculated output size: ($nOutputPlane x $outputHeight x $outputWidth). " +
        s"Output size is too small")

    require(input.dim() == nDim && input.size(dimF) == nInputPlane)

    if (null != gradOutput) {
      require(gradOutput.nDimension() == nDim &&
        gradOutput.size(dimF) == nOutputPlane &&
        gradOutput.size(dimH) == outputHeight &&
        gradOutput.size(dimW) == outputWidth
      )
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"AtrousConvolution2D requires 4D input, but got input dim ${input.length}")
    val outputWidth = (input(3) + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1
    val outputHeight = (input(2) + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1
    Shape(input(0), nOutputPlane, outputHeight, outputWidth)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    shapeCheck(input, null, weight, bias,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW)
    require(input.isContiguous())

    val isBatch = if (input.nDimension() == 3) {
      // Force batch
      input.resize(1, input.size(1), input.size(2), input.size(3))
      false
    } else {
      true
    }

    val inputWidth = input.size(4)
    val inputHeight = input.size(3)
    val outputWidth = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1
    val outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1

    // Batch size + input planes
    val batchSize = input.size(1)

    // Resize output
    output.resize(batchSize, nOutputPlane, outputHeight, outputWidth)
    output.zero()

    if (null == fInput) {
      fInput = Tensor[T]()
    }
    // Resize temporary columns
    val columns = fInput
    columns.resize(nInputPlane*kW*kH, outputHeight*outputWidth)

    if (null == fGradInput) {
      fGradInput = Tensor[T]()
    }
    // Define a buffer of ones, for bias accumulation
    val ones = fGradInput
    if (ones.nDimension != 2 || ones.size(1)*ones.size(2) < outputHeight*outputWidth) {
      // Resize plane and fill with ones...
      ones.resize(outputHeight, outputWidth)
      ones.fill(ev.fromType[Int](1))
    }

    // For each element in batch, do:
    var elt = 1
    while (elt <= batchSize) {
      // Matrix multiply per output:
      val input_n = input.select(1, elt)
      val output_n = output.select(1, elt)

      // Do Bias first:
      // M,N,K are dims of matrix A and B
      var m = nOutputPlane
      var n = outputHeight * outputWidth
      var k = 1

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      if (null != bias) {
        DenseTensorBLAS.gemm[T](
          't', 'n',
          n, m, k,
          ev.fromType[Int](1),
          ones.storage().array(), ones.storageOffset() - 1, k,
          bias.storage().array(), bias.storageOffset() - 1, k,
          ev.fromType[Int](0),
          output_n.storage().array(), output_n.storageOffset() - 1, n
        )
      } else {
        output_n.zero()
      }

      // Extract columns:
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.im2colWithDilationDouble(
          input_n.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
          nInputPlane, inputHeight, inputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          dilationH, dilationW
        )
        case FloatType => NNPrimitive.im2colWithDilationFloat(
          input_n.asInstanceOf[Tensor[Float]], columns.asInstanceOf[Tensor[Float]],
          nInputPlane, inputHeight, inputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          dilationH, dilationW
        )
        case t => throw new NotImplementedError(s"$t is not supported")
      }
      im2colTime += System.nanoTime() - before

      // M,N,K are dims of matrix A and B
      m = nOutputPlane
      n = columns.size(2)
      k = nInputPlane*kH*kW

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        'n', 'n',
        n, m, k,
        ev.fromType[Int](1),
        columns.storage().array(), columns.storageOffset() - 1, n,
        weight.storage().array(), weight.storageOffset() - 1, k,
        ev.fromType[Int](1),
        output_n.storage().array(), output_n.storageOffset() - 1, n
      )
      elt += 1
    }

    // Resize output
    if (!isBatch) {
      output.resize(nOutputPlane, outputHeight, outputWidth)
      input.resize(nInputPlane, inputHeight, inputWidth)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    shapeCheck(input, gradOutput, weight, null,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW)

    val isBatch = if (input.nDimension() == 3) {
      // Force batch
      input.resize(1, input.size(1), input.size(2), input.size(3))
      gradOutput.resize(1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3))
      false
    } else {
      true
    }

    val inputWidth = input.size(4)
    val inputHeight = input.size(3)
    val outputWidth = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1
    val outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1

    // Batch size + input planes
    val batchSize = input.size(1)

    // Resize output
    gradInput.resize(batchSize, nInputPlane, inputHeight, inputWidth).zero()

    // Resize temporary columns
    val gradColumns = fInput
    gradColumns.resize(nInputPlane*kW*kH, outputHeight*outputWidth);
    gradColumns.zero()

    // For each element in batch, do:
    var elt = 1
    while (elt <= batchSize) {
      // Matrix multiply per sample:
      val gradInput_n = gradInput.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)

      // M,N,K are dims of matrix A and B
      val m = nInputPlane*kW*kH
      val n = gradColumns.size(2)
      val k = nOutputPlane

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        'n', 't',
        n, m, k,
        ev.fromType[Int](1),
        gradOutput_n.storage().array(), gradOutput_n.storageOffset() - 1, n,
        weight.storage().array(), weight.storageOffset() - 1, m,
        ev.fromType[Int](0),
        gradColumns.storage().array(), gradColumns.storageOffset() - 1, n
      )

      // Unpack columns back into input:
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.col2imWithDilationDouble(
          gradColumns.asInstanceOf[Tensor[Double]], gradInput_n.asInstanceOf[Tensor[Double]],
          nInputPlane, inputHeight, inputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          dilationH, dilationW
        )
        case FloatType => NNPrimitive.col2imWithDilationFloat(
          gradColumns.asInstanceOf[Tensor[Float]], gradInput_n.asInstanceOf[Tensor[Float]],
          nInputPlane, inputHeight, inputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          dilationH, dilationW
        )
        case t => throw new NotImplementedError(s"$t is not supported")
      }
      col2imTime += System.nanoTime() - before
      elt += 1
    }

    // Resize output
    if (!isBatch) {
      gradOutput.resize(nOutputPlane, outputHeight, outputWidth)
      input.resize(nInputPlane, inputHeight, inputWidth)
      gradInput.resize(nInputPlane, inputHeight, inputWidth)
    }

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    shapeCheck(input, gradOutput, gradWeight, gradBias,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW)

    val isBatch = if (input.nDimension() == 3) {
      // Force batch
      input.resize(1, input.size(1), input.size(2), input.size(3))
      gradOutput.resize(1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3))
      false
    } else {
      true
    }

    val inputWidth = input.size(4)
    val inputHeight = input.size(3)
    val outputWidth = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1
    val outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1

    // Batch size + input planes
    val batchSize = input.size(1)

    // Define a buffer of ones, for bias accumulation
    val ones = fGradInput
    if (ones.nDimension != 2 || ones.size(1)*ones.size(2) < outputHeight*outputWidth) {
      // Resize plane and fill with ones...
      ones.resize(outputHeight, outputWidth)
      ones.fill(ev.fromType[Int](1))
    }

    // Resize temporary columns
    val columns = fInput
    columns.resize(nInputPlane*kW*kH, outputHeight*outputWidth)

    // For each element in batch, do:
    var elt = 1
    while (elt <= batchSize) {
      // Matrix multiply per output:
      val input_n = input.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)

      // Extract columns:
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.im2colWithDilationDouble(
          input_n.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
          nInputPlane, inputHeight, inputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          dilationH, dilationW
        )
        case FloatType => NNPrimitive.im2colWithDilationFloat(
          input_n.asInstanceOf[Tensor[Float]], columns.asInstanceOf[Tensor[Float]],
          nInputPlane, inputHeight, inputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          dilationH, dilationW
        )
        case t => throw new NotImplementedError(s"$t is not supported")
      }
      im2colTime += System.nanoTime() - before

      // M,N,K are dims of matrix A and B
      var m = nOutputPlane
      val n = nInputPlane*kW*kH
      var k = columns.size(2)

      if (scaleW != 0) {
        // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
        DenseTensorBLAS.gemm[T](
          't', 'n',
          n, m, k,
          ev.fromType[Double](scaleW),
          columns.storage().array(), columns.storageOffset() - 1, k,
          gradOutput_n.storage().array(), gradOutput_n.storageOffset() - 1, k,
          ev.fromType[Int](1),
          gradWeight.storage().array(), gradWeight.storageOffset() - 1, n
        )
      }


      // Do Bias:
      // M,N,K are dims of matrix A and B
      m = nOutputPlane
      k = outputHeight * outputWidth

      // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
      if (null != gradBias && scaleB != 0) {
        ev.gemv(
          't',
          k, m,
          ev.fromType[Double](scaleB),
          gradOutput_n.storage().array(), gradOutput_n.storageOffset() - 1, k,
          ones.storage().array(), ones.storageOffset() - 1, 1,
          ev.fromType[Int](1),
          gradBias.storage().array(), gradBias.storageOffset() - 1, 1
        )
      }
      elt += 1
    }

    // Resize
    if (!isBatch) {
      gradOutput.resize(nOutputPlane, outputHeight, outputWidth)
      input.resize(nInputPlane, inputHeight, inputWidth)
    }

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialDilatedConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialDilatedConvolution[T]]
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
      dilationW == other.dilationW &&
      dilationH == other.dilationH &&
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
    hash = hash * seed + dilationW.hashCode()
    hash = hash * seed + dilationH.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane -> $nOutputPlane, " +
      s"$kW x $kH, $dW, $dH, $padW, $padH, $dilationH, $dilationW)"
  }
}

object SpatialDilatedConvolution extends quantized.Quantizable {
  def apply[@specialized(Float, Double) T: ClassTag](
      nInputPlane: Int,
      nOutputPlane: Int,
      kW: Int,
      kH: Int,
      dW: Int = 1,
      dH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      dilationW: Int = 1,
      dilationH: Int = 1,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]) : SpatialDilatedConvolution[T] = {
    new SpatialDilatedConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH,
      wRegularizer, bRegularizer)
  }
  def quantize[T: ClassTag](module: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val conv = module.asInstanceOf[SpatialDilatedConvolution[T]]
    quantized.SpatialDilatedConvolution[T](
      conv.nInputPlane, conv.nOutputPlane, conv.kW, conv.kH, conv.dW,
      conv.dH, conv.padW, conv.padH, conv.dilationW, conv.dilationW, initWeight = conv.weight,
      initBias = conv.bias).setName(conv.getName())
  }
}
