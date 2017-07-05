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

import com.intel.analytics.bigdl.fixpoint.FixPoint
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.nn.{ErrorInfo, VariableFormat, SpatialConvolution => NNSpatialConvolution}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils._

import scala.reflect.ClassTag

@SerialVersionUID(- 8008252944905538960L)
class SpatialConvolution[T: ClassTag](
  nInputPlane: Int, // The number of expected input planes in the image given into forward()
  nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  kernelW: Int, // The kernel width of the convolution
  kernelH: Int, // The kernel height of the convolution
  strideW: Int = 1, // The step of the convolution in the width dimension.
  strideH: Int = 1, // The step of the convolution in the height dimension
  padW: Int = 0, // The additional zeros added per width to the input planes.
  padH: Int = 0, // The additional zeros added per height to the input planes.
  nGroup: Int = 1 // Kernel group number
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  var weight = 0L
  var weightSum = 0L
  var data = 0L

  val bias = Tensor[T](nOutputPlane)
  val weightSumFP32 = Tensor() // TODO

  val FAULT_TOLERANCE = 0.5f
  var THRESHOLD = 127.0f
  val DILATION_HEIGHT = 1
  val DILATION_WIDTH = 1

  @transient var _init = false
  def init(weightFP32: Tensor[Float]): this.type = {
    weight = FixPoint.FixConvKernelDescInit(nOutputPlane, nInputPlane, kernelH, kernelW)
    FixPoint.FixConvKernelInit(weight, weightFP32.storage().array(),
      weightFP32.storageOffset() - 1,
      nOutputPlane, nInputPlane, kernelH, kernelW, 64.0f, FixPoint.NCHW)
    weightSum = FixPoint.FixConvKernelSumDescInit(nOutputPlane)
    FixPoint.FixConvKernelSumInit(weightSum, weightFP32.storage().array(),
      weightFP32.storageOffset() - 1,
      nOutputPlane, nInputPlane, kernelH, kernelW)
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    FixPoint.printHello()

    val batchSize = input.size(1)

    // TODO if the input size has changed, we should renew one
    if (data == 0) {
      data = FixPoint.FixConvDataDescInit(
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
        input.size(2),
        input.size(3))
    }

    FixPoint.FixConvDataInit(data, input.storage().array().asInstanceOf[Array[Float]],
      input.storageOffset() - 1,
      nInputPlane,
      kernelH,
      kernelW,
      strideH,
      strideW,
      padH,
      padW,
      DILATION_HEIGHT,
      DILATION_WIDTH,
      input.size(1),
      input.size(2),
      input.size(3),
      THRESHOLD,
      FixPoint.NCHW)

    FixPoint.InternalMixPrecisionConvolutionGEMM(
      FixPoint.NCHW,
      weight, data,
      output.storage().array().asInstanceOf[Array[Float]], output.storageOffset() - 1,
      nOutputPlane, batchSize, nInputPlane,
      weightSum,
      bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1,
      input.size(1), input.size(2)/nGroup, output.size(3), output.size(4),
      FAULT_TOLERANCE)

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
      propagateBack == other.propagateBack &&
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
    hash = hash * seed + weight.hashCode()
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
  }
}

object SpatialConvolution {
}

object TestFPConv {
  def main(args: Array[String]): Unit = {
    val conv = new SpatialConvolution[Float](1, 1, 3, 3, 1, 1, 2, 2, 1)
    val input = Tensor[Float]()

    conv.updateOutput(input)
  }
}

