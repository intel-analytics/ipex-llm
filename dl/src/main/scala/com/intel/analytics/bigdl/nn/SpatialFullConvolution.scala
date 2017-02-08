/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, AbstractModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.reflect.ClassTag

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
 * @param noBias If bias is needed.
 * @param initMethod Init method, Default, Xavier, Bilinear.
 */

@SerialVersionUID(- 3110412775551642284L)
class SpatialFullConvolution[A <: Activity : ClassTag, T: ClassTag](
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
  val noBias: Boolean = false,
  private var initMethod: InitializationMethod = Default
  )(implicit ev: TensorNumeric[T]) extends AbstractModule[A, Tensor[T], T]{

  require(adjW <= dW - 1 && adjH <= dH - 1,
    "adjW and adjH must be smaller than dW - 1 and dH - 1 respectively")

  val weight: Tensor[T] = Tensor[T](nInputPlane, nOutputPlane, kH, kW)
  val bias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)

  val gradWeight: Tensor[T] = Tensor[T](nInputPlane, nOutputPlane, kH, kW)
  val gradBias: Tensor[T] = if (noBias) null else Tensor[T](nOutputPlane)
  @transient private var columns: Tensor[T] = null
  @transient private var ones: Tensor[T] = null
  @transient private var zeroScalar: Tensor[T] = null

  reset()

  private var im2colTime = 0L
  private var col2imTime = 0L

  def getIm2ColTime(): Double = im2colTime

  def getCol2ImgTime(): Double = col2imTime

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(kW * kH * nInputPlane)
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        if (!noBias) {
          bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        }
      case Xavier =>
        val fanIn = nInputPlane * kH * kW
        val fanOut = nOutputPlane * kH * kW
        val stdv = math.sqrt(6.0 / (fanIn + fanOut))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
        if (null != bias) {
          bias.fill(ev.fromType(0))
        }
      case BilinearFiller =>
        require(weight.nDimension() == 4, "weight must be 4 dim")
        require(kH == kW, "Kernel must be square")
        val f = Math.ceil(kW / 2.0).toInt
        val c = (2 * f - 1 - f % 2) / (2.0f * f)
        val weightArray = weight.storage().array()
        val weightOffset = weight.storageOffset() - 1
        var i = 0
        while(i < weight.nElement()) {
          val x : Float = i % kW
          val y : Float = (i / kW) % kH
          weightArray(i + weightOffset) = ev.fromType[Float](
            (1f - math.abs(x / f - c)) * (1f - math.abs(y / f - c)))
          i += 1
        }
    }
    zeroGradParameters()
  }

  private def calculateAdj(targetSize : Int, ker : Int, pad : Int, stride : Int) : Int = {
    return (targetSize + 2 * pad - ker) % stride
  }

  private def shapeCheck(input : Tensor[T], gradOutput : Tensor[T],
    weight : Tensor[T], bias : Tensor[T],
    kH : Int, kW : Int,
    dH : Int, dW : Int,
    padH : Int, padW : Int,
    adjH : Int, adjW : Int) : Unit = {

    require(kW > 0 && kH > 0, s"kernel size should be greater than zero, but got kH: $kH kW: $kW")
    require(dW > 0 && dH > 0, s"stride should be greater than zero, but got dH: $dH dW: $dW")
    require(weight.nDimension == 2 || weight.nDimension == 4,
      s"2D or 4D weight tensor expected, but got size: ${weight.size()}")

    if (null != bias) {
      require(bias.nDimension() == 1 && bias.size(1) == weight.size(2))
    }

    val ndim = input.nDimension
    val dimf = if (ndim == 4) 2 else 1
    val dimh = if (ndim == 4) 3 else 2
    val dimw = if (ndim == 4) 4 else 3

    require(ndim == 3 || ndim == 4, s"3D or 4D input tensor expected but got size: ${input.size()}")

    val inputHeight = input.size(dimh)
    val inputWidth = input.size(dimw)
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW

    require(outputWidth >= 1 || outputHeight >= 1,
      s"Given input size: ($nInputPlane x $inputHeight x $inputWidth). " +
      s"Calculated output size: ($nOutputPlane x $outputHeight x $outputWidth). " +
      s"Output size is too small")

    require(input.nDimension() == ndim && input.size(dimf) == nInputPlane)

    if (null != gradOutput) {
      require(gradOutput.nDimension() == ndim && gradOutput.size(dimf) == nOutputPlane)
      require(gradOutput.nDimension() == ndim && gradOutput.size(dimh) == outputHeight)
      require(gradOutput.nDimension() == ndim && gradOutput.size(dimw) == outputWidth)
    }
  }

  override def updateOutput(input: A): Tensor[T] = {
    val inputTensor: Tensor[T] = if (input.isInstanceOf[Table]) {
      val targetTensor: Tensor[T] = input.toTable[Tensor[T]](2)
      val tDims = targetTensor.dim()
      val tH = targetTensor.size(tDims - 1)
      val tW = targetTensor.size(tDims)
      adjW = calculateAdj(tW, kW, padW, dW)
      adjH = calculateAdj(tH, kH, padH, dH)
      input.toTable[Tensor[T]](1)
    } else {
      input.toTensor[T]
    }


    shapeCheck(inputTensor, null, weight, bias, kH, kW, dH, dW, padH, padW, adjH, adjW)
    require(inputTensor.isContiguous())

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

    // Resize temporary columns
    if(null == columns) {
      columns = Tensor[T]()
    }
    columns.resize(nOutputPlane * kW * kH, inputHeight * inputWidth)
    columns.zero()

    // Define a buffer of ones, for bias accumulation
    // Note: this buffer can be shared with other modules, it only ever gets increased,
    // and always contains ones.
    if(null == ones) {
      ones = Tensor[T]()
    }
    if (ones.nDimension != 2 || ones.size(1) * ones.size(2) < outputHeight * outputWidth) {
      // Resize plane and fill with ones...
      ones.resize(outputHeight, outputWidth)
      ones.fill(ev.fromType[Int](1))
    }

    var elt = 1
    // For each element in batch, do:
    while(elt <= batchSize) {
      // Matrix mulitply per output:
      val input_n = inputTensor.select(1, elt)
      val output_n = output.select(1, elt)

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      var m = weight.size(2) * weight.size(3) * weight.size(4)
      var n = columns.size(2)
      var k = weight.size(1)

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        'N', 'T',
        n, m, k,
        ev.fromType[Int](1),
        input_n.storage().array(), input_n.storageOffset() - 1, n,
        weight.storage().array(), weight.storageOffset() - 1, m,
        ev.fromType[Int](0),
        columns.storage().array(), columns.storageOffset() - 1, n
      )

      // Unpack columns back into input:
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.col2imWithDilationDouble(
          columns.asInstanceOf[Tensor[Double]], output_n.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case FloatType => NNPrimitive.col2imWithDilationFloat(
          columns.asInstanceOf[Tensor[Float]], output_n.asInstanceOf[Tensor[Float]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )
      }
      col2imTime += System.nanoTime() - before

      // Do Bias after:
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      m = nOutputPlane
      n = outputHeight * outputWidth
      k = 1

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      if(null != bias) {
        DenseTensorBLAS.gemm[T](
          'T', 'N',
          n, m, k,
          ev.fromType[Int](1),
          ones.storage().array(), ones.storageOffset() - 1, k,
          bias.storage().array(), bias.storageOffset() - 1, k,
          ev.fromType[Int](1),
          output_n.storage().array(), output_n.storageOffset() - 1, n
        )
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

  override def updateGradInput(input: A, gradOutput: Tensor[T]): A = {
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

    gradInputTensor.resize(batchSize, nInputPlane, inputHeight, inputWidth)
    gradInputTensor.zero()

    columns.resize(nOutputPlane * kW * kH, inputHeight * inputWidth)

    var elt = 1
    // For each element in batch, do:
    while (elt <= batchSize) {
      // Matrix mulitply per sample:
      val gradInput_n = gradInputTensor.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)

      // Extract columns:
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.im2colWithDilationDouble(
          gradOutput_n.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case FloatType => NNPrimitive.im2colWithDilationFloat(
          gradOutput_n.asInstanceOf[Tensor[Float]], columns.asInstanceOf[Tensor[Float]],
          nOutputPlane, outputHeight,
          outputWidth, kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )
      }
      im2colTime += System.nanoTime() - before

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      val m = weight.size(1)
      val n = columns.size(2)
      val k = weight.size(2) * weight.size(3) * weight.size(4)

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        'N', 'N',
        n, m, k,
        ev.fromType[Int](1),
        columns.storage().array(), columns.storageOffset() - 1, n,
        weight.storage().array(), weight.storageOffset() - 1, k,
        ev.fromType[Int](0),
        gradInput_n.storage().array(), gradInput_n.storageOffset() - 1, n
      )
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
      if (null == zeroScalar) zeroScalar = input2.clone().zero()
      ones.resizeAs(input2).fill(ev.fromType[Int](1))
      val zeroTensor = zeroScalar.view(ones.size()).expandAs(input2)
      gradInput.toTable(1) = gradInputTensor
      gradInput.toTable(2) = zeroTensor
    }

    return gradInput
  }

  override def accGradParameters(input: A, gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
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

    // Define a buffer of ones, for bias accumulation
    if (ones.nDimension != 2 || ones.size(1) * ones.size(2) < outputHeight * outputWidth) {
      // Resize plane and fill with ones...
      ones.resize(outputHeight, outputWidth)
      ones.fill(ev.fromType[Int](1))
    }

    // Resize temporary columns
    columns.resize(nOutputPlane * kW * kH, inputHeight * inputWidth)

    var elt = 1
    // For each element in batch, do:
    while (elt <= batchSize) {
      // Matrix multiply per output:
      val input_n = inputTensor.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)

      // Extract columns:
      val before = System.nanoTime()
      ev.getType() match {
        case DoubleType => NNPrimitive.im2colWithDilationDouble(
          gradOutput_n.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case FloatType => NNPrimitive.im2colWithDilationFloat(
          gradOutput_n.asInstanceOf[Tensor[Float]], columns.asInstanceOf[Tensor[Float]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )
      }
      im2colTime += System.nanoTime() - before

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      val n = columns.size(1)   // nOutputPlane * kh * kw
      var m = input_n.size(1)   // nInputPlane
      var k = columns.size(2)   // inputHeight * inputWidth

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        'T', 'N',
        n, m, k,
        ev.fromType[Double](scale),
        columns.storage().array(), columns.storageOffset() - 1, k,
        input_n.storage().array(), input_n.storageOffset() - 1, k,
        ev.fromType[Int](1),
        gradWeight.storage().array(), gradWeight.storageOffset() - 1, n
      )

      // Do Bias:
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      m = nOutputPlane
      k = outputHeight * outputWidth

      // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
      if (null != gradBias) {
        ev.gemv(
          'T',
          k, m,
          ev.fromType[Double](scale),
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
      inputTensor.resize(nInputPlane, inputHeight, inputWidth)
    }

  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    if(!noBias) {
      gradBias.zero()
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialFullConvolution[A, T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialFullConvolution[A, T]]
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
    s"nn.SpatialFullConvolution($nInputPlane -> $nOutputPlane, " +
      s"$kW x $kH, $dW, $dH, $padW, $padH, $adjW, $adjH)"
  }
}

object SpatialFullConvolution {
  def apply[A <: Activity : ClassTag, @specialized(Float, Double) T: ClassTag](
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
      noBias: Boolean = false,
      initMethod: InitializationMethod = Default
  )(implicit ev: TensorNumeric[T]) : SpatialFullConvolution[A, T] = {
    new SpatialFullConvolution[A, T](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, adjW, adjH, noBias, initMethod)
  }
}
