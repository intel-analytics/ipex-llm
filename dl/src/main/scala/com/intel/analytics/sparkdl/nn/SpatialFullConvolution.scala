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

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.Activities
import com.intel.analytics.sparkdl.utils.RandomGenerator._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag

class SpatialFullConvolution[@specialized(Float, Double) T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kW: Int, // The kernel width of the convolution
  val kH: Int, // The kernel height of the convolution
  val dW: Int = 1, // The step of the convolution in the width dimension.
  val dH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  val adjW: Int = 0, // Extra width to add to the output image.
  val adjH: Int = 0, // Extra height to add to the output image.
  private var initMethod: InitializationMethod = Default
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(adjW <= dW - 1 && adjH <= dH - 1,
    "adjW and adjH must be smaller than dW - 1 and dH - 1 respectively")

  val weight: Tensor[T] = Tensor[T](nInputPlane, nOutputPlane, kH, kW)
  this.gradWeight = Tensor[T](nInputPlane, nOutputPlane, kH, kW)

  val bias: Tensor[T] = Tensor[T](nOutputPlane)
  this.gradBias = Tensor[T](nOutputPlane)
  @transient
  var columns : Tensor[T] = null
  @transient
  var ones : Tensor[T] = null
  reset()

  private var im2colTime = 0L
  private var col2imTime = 0L

  def getIm2ColTime(): Double = im2colTime

  def getCol2ImgTime(): Double = col2imTime

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
      case Bilinear =>
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

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    shapeCheck(input, null, weight, bias, kH, kW, dH, dW, padH, padW, adjH, adjW)
    require(input.isContiguous())

    val batch = if (input.nDimension() == 3) {
      // Force batch
      input.resize(1, input.size(1), input.size(2), input.size(3))
      0
    } else {
      1
    }

    val inputWidth = input.size(3)
    val inputHeight = input.size(4)

    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW

    // Batch size + input planes
    val batchSize = input.size(1)

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
    // For each elt in batch, do:
    while(elt <= batchSize) {
      // Matrix mulitply per output:
      val input_n = input.select(1, elt)
      val output_n = output.select(1, elt)

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      var m = weight.size(2) * weight.size(3) * weight.size(4)
      var n = columns.size(2)
      var k = weight.size(1)

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      DenseTensorBLAS.gemm[T](
        "N", "T",
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
        case "Double" => NNPrimitive.col2imWithDilationDouble(
          columns.asInstanceOf[Tensor[Double]], output_n.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case "Float" => NNPrimitive.col2imWithDilationFloat(
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
          "T", "N",
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
    if(batch == 0) {
      output.resize(nOutputPlane, outputHeight, outputWidth)
      input.resize(nInputPlane, inputHeight, inputWidth)
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    shapeCheck(input, gradOutput, weight, null, kH, kW, dH, dW, padH, padW, adjH, adjW)

    val batch = if (input.nDimension() == 3) {
      // Force batch
      input.resize(1, input.size(1), input.size(2), input.size(3))
      gradOutput.resize(1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3))
      0
    } else {
      1
    }

    val inputWidth = input.size(4)
    val inputHeight = input.size(3)
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH

    // Batch size + input planes
    val batchSize = input.size(1)

    gradInput.resize(batchSize, nInputPlane, inputHeight, inputWidth)
    gradInput.zero()

    columns.resize(nOutputPlane * kW * kH, inputHeight * inputWidth)

    var elt = 1
    // For each elt in batch, do:
    while (elt <= batchSize) {
      // Matrix mulitply per sample:
      val gradInput_n = gradInput.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)

      // Extract columns:
      val before = System.nanoTime()
      ev.getType() match {
        case "Double" => NNPrimitive.im2colWithDilationDouble(
          gradOutput_n.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case "Float" => NNPrimitive.im2colWithDilationFloat(
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
        "N", "N",
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
    if (batch == 0) {
      gradOutput.resize(nOutputPlane, outputHeight, outputWidth)
      input.resize(nInputPlane, inputHeight, inputWidth)
      gradInput.resize(nInputPlane, inputHeight, inputWidth)
    }

    return gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    shapeCheck(input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW, adjH, adjW)

    val batch = if (input.nDimension() == 3) {
      // Force batch
      input.resize(1, input.size(1), input.size(2), input.size(3))
      gradOutput.resize(1, gradOutput.size(1), gradOutput.size(2), gradOutput.size(3))
      0
    } else {
      1
    }

    val inputWidth = input.size(4)
    val inputHeight = input.size(3)
    val outputWidth = (inputWidth - 1) * dW - 2 * padW + kW + adjW
    val outputHeight = (inputHeight - 1) * dH - 2 * padH + kH + adjH

    // Batch size + input planes
    val batchSize = input.size(1)

    // Define a buffer of ones, for bias accumulation
    if (ones.nDimension != 2 || ones.size(1) * ones.size(2) < outputHeight * outputWidth) {
      // Resize plane and fill with ones...
      ones.resize(outputHeight, outputWidth)
      ones.fill(ev.fromType[Int](1))
    }

    // Resize temporary columns
    columns.resize(nOutputPlane * kW * kH, inputHeight * inputWidth)

    var elt = 1
    // For each elt in batch, do:
    while (elt <= batchSize) {
      // Matrix mulitply per output:
      val input_n = input.select(1, elt)
      val gradOutput_n = gradOutput.select(1, elt)

      // Extract columns:
      val before = System.nanoTime()
      ev.getType() match {
        case "Double" => NNPrimitive.im2colWithDilationDouble(
          gradOutput_n.asInstanceOf[Tensor[Double]], columns.asInstanceOf[Tensor[Double]],
          nOutputPlane, outputHeight, outputWidth,
          kH, kW,
          padH, padW,
          dH, dW,
          1, 1
        )

        case "Float" => NNPrimitive.im2colWithDilationFloat(
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
        "T", "N",
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
          "T",
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
    if (batch == 0) {
      gradOutput.resize(nOutputPlane, outputHeight, outputWidth)
      input.resize(nInputPlane, inputHeight, inputWidth)
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
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.SpatialFullConvolution($nInputPlane -> $nOutputPlane, $kW x $kH, $dW, $dH, $padW, $padH)"
  }

  override def findModel(paramOffset: Int,
                         indexes: Array[Int]): (Module[_ <: Activities, _ <: Activities, T], Int, Array[Int]) = {
    (this, paramOffset - nOutputPlane * nInputPlane * kH * kW - nOutputPlane, indexes)
  }
}
