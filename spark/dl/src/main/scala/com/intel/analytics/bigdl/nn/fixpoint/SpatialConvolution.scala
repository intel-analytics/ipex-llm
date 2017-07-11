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

package com.intel.analytics.bigdl.nn.fixpoint

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.{QuantizeTensor, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{ErrorInfo, Module, Quantize}
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.fixpoint.FixPoint

import scala.reflect.ClassTag

@SerialVersionUID(- 8008252944905538960L)
class SpatialConvolution[T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kernelW: Int, // The kernel width of the convolution
  val kernelH: Int, // The kernel height of the convolution
  val strideW: Int = 1, // The step of the convolution in the width dimension.
  val strideH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  val nGroup: Int = 1 // Kernel group number
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  val FIX_TENSOR = 0
  val FP_TENSOR = 1

  var weight: QuantizeTensor[T] = QuantizeTensor[T](FIX_TENSOR)
  var data: QuantizeTensor[T] = QuantizeTensor[T](FIX_TENSOR)

  val bias: Tensor[T] = Tensor[T](nOutputPlane)
  var weightSum: Array[T] = _

  val FAULT_TOLERANCE = 0.5f
  val WEIGHT_THRESHOLD = 64.0f
  val THRESHOLD = 127.0f
  val DILATION_HEIGHT = 1
  val DILATION_WIDTH = 1

  val min = new Array[T](nInputPlane * kernelH * kernelW)
  val max = new Array[T](nInputPlane * kernelH * kernelW)

  @transient var _init = false

  private def getWeightSum(weight: Tensor[T]): Array[T] = {
    val array = new Array[T](nOutputPlane)
    for (i <- 1 to nOutputPlane) {
      val singleRow = weight.select(1, i)
      array(i - 1) = singleRow.sum()
    }

    array
  }

  def init(weightFP32: Tensor[T], biasFP32: Tensor[T]): this.type = {
    weight.setStorageInJni(
      FixPoint.FixConvKernelDescInit(nOutputPlane, nInputPlane, kernelH, kernelW))

    bias.copy(biasFP32)

    val weightFP32Tmp = weightFP32.view(Array(nOutputPlane, nInputPlane, kernelH, kernelW))
    weightSum = getWeightSum(weightFP32Tmp)

    for (i <- 1 to nOutputPlane) {
      val singleRow = weightFP32Tmp.select(1, i)
      min(i - 1) = singleRow.min()
      max(i - 1) = singleRow.max()
    }

    val buffer = ByteBuffer.allocate(weightFP32.nElement())
    Quantize.quantize(weightFP32.asInstanceOf[Tensor[Float]], buffer, 0)

    weight.setBufferFromInterStorage(buffer)

    val byteBuffer = buffer.array()
    val charArray = new Array[Char](buffer.remaining())

    FixPoint.FixConvKernelLoadFromModel(
      weight.getStorageInJni, byteBuffer, 0,
      min.asInstanceOf[Array[Float]], max.asInstanceOf[Array[Float]], nOutputPlane, nInputPlane,
      kernelH, kernelW, WEIGHT_THRESHOLD, FixPoint.NCHW)

    this
  }

  private def outputSize(input: Int, pad: Int, kernel: Int, stride: Int): Int = {
    (input + 2 * pad - kernel) / stride + 1
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    FixPoint.printHello()

    if (!_init && weight.getBufferFromInterStorage.isDefined) {
      val byteArrayOfWeight = weight.getBufferFromInterStorage.get
      weight.setStorageInJni(
        FixPoint.FixConvKernelDescInit(nOutputPlane, nInputPlane, kernelH, kernelW))

      FixPoint.FixConvKernelLoadFromModel(
        weight.getStorageInJni, byteArrayOfWeight, 0,
        min.asInstanceOf[Array[Float]], max.asInstanceOf[Array[Float]], nOutputPlane, nInputPlane,
        kernelH, kernelW, WEIGHT_THRESHOLD, FixPoint.NCHW)
    }

    val (batchSize, inputHeight, inputWidth) = (input.size(1), input.size(3), input.size(4))
    val outputHeight = outputSize(inputHeight, padH, kernelH, strideH)
    val outputWidth = outputSize(inputWidth, padW, kernelW, strideW)

    // TODO 3-D
    output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))

    // TODO if the input size has changed, we should renew one
    if (!data.isInitialized) {
      data.setStorageInJni(FixPoint.FixConvDataDescInit(
        nInputPlane,
        kernelH,
        kernelW,
        strideH,
        strideW,
        padH,
        padW,
        DILATION_HEIGHT,
        DILATION_WIDTH,
        batchSize,
        inputHeight,
        inputWidth))
    }

    FixPoint.FixConvDataInit(
      data.getStorageInJni,
      input.storage().array().asInstanceOf[Array[Float]], input.storageOffset() - 1,
      nInputPlane,
      kernelH,
      kernelW,
      strideH,
      strideW,
      padH,
      padW,
      DILATION_HEIGHT,
      DILATION_WIDTH,
      batchSize,
      inputHeight,
      inputWidth,
      THRESHOLD,
      FixPoint.NCHW)

    var i = 1
    while (i <= batchSize) {
      val outputT = output.select(1, i)
      FixPoint.InternalMixPrecisionConvolutionGEMM(
        FixPoint.NCHW,
        weight.getStorageInJni, data.getStorageInJni,
        outputT.storage().array().asInstanceOf[Array[Float]], outputT.storageOffset() - 1,
        nOutputPlane, batchSize, nInputPlane,
        weightSum.asInstanceOf[Array[Float]],
        bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1,
        input.size(1), input.size(2) / nGroup, output.size(3), output.size(4),
        FAULT_TOLERANCE)
      i += 1
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
  }

  override def updateParameters(learningRate: T): Unit = {
  }

  override def zeroGradParameters(): Unit = {
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(Tensor(), Tensor()), Array(Tensor(), Tensor()))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "bias" -> bias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if (this.eq(other)) {
      return true
    }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kernelW == other.kernelW &&
      kernelH == other.kernelH &&
      strideW == other.strideW &&
      strideH == other.strideH &&
      padW == other.padW &&
      padH == other.padH &&
      nGroup == other.nGroup &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kernelW.hashCode()
    hash = hash * seed + kernelH.hashCode()
    hash = hash * seed + strideW.hashCode()
    hash = hash * seed + strideH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"fixpoint.SpatialConvolution($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH)"
  }

  def release(): Unit = {
    weight.release()
    data.release()
  }
}

object TestFPConv {
  def main(args: Array[String]): Unit = {
    import com.intel.analytics.bigdl.nn.{SpatialConvolution => NNSpatialConvolution}
    val test = TestCase(2, 512, 10, 10, 1, 126, 3, 3, 1, 1, 1, 1)

    val weight = Tensor[Float](test.group, test.outputChannel / test.group,
      test.inputChannel / test.group, test.kernelHeight, test.kernelWidth).fill(1.0f)
    val bias = Tensor[Float](test.outputChannel).fill(0f)

    val nnConv = new NNSpatialConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group, initWeight = weight, initBias = bias)

    val input = Tensor[Float]().resize(Array(test.batchSize, test.inputChannel,
      test.inputHeight, test.inputWidth)).fill(1.0f)

    val quantizedConv = new SpatialConvolution[Float](test.inputChannel, test.outputChannel,
      test.kernelHeight, test.kernelWidth, test.strideHeight, test.strideWidth,
      test.padHeight, test.padWidth, test.group)

    nnConv.updateOutput(input)
    quantizedConv.init(nnConv.weight, nnConv.bias)
    quantizedConv.updateOutput(input)

    println(nnConv.output)
    println(quantizedConv.output)

    quantizedConv.release()
    Files.deleteIfExists(Paths.get("/tmp/quantizedConv"))
    quantizedConv.save("/tmp/quantizedConv")
    val tmp = Module.load("/tmp/quantizedConv").asInstanceOf[SpatialConvolution[Float]]
    println(tmp)
    tmp.updateOutput(input)
    nnConv.save("/tmp/nnConv")
//    println("="*80)
//    println(tmp.output)
//    println("="*80)
  }
}

