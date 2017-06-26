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
import com.intel.analytics.bigdl.nn.{ErrorInfo, VariableFormat, SpatialConvolution => NNSpatialConvolution}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.RandomGenerator._

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
  nGroup: Int = 1, // Kernel group number
  propagateBack: Boolean = false, // propagate gradient back
  wRegularizer: Regularizer[T] = null,
  bRegularizer: Regularizer[T] = null,
  initWeight: Tensor[T] = null,
  initBias: Tensor[T] = null,
  initGradWeight: Tensor[T] = null,
  initGradBias: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends NNSpatialConvolution[T](
  nInputPlane,
  nOutputPlane,
  kernelW,
  kernelH,
  strideW,
  strideH,
  padW,
  padH,
  nGroup,
  propagateBack,
  wRegularizer,
  bRegularizer,
  initWeight,
  initBias,
  initGradWeight,
  initGradBias
) {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  reset()

  @transient
  var _init = false
  @transient
  var desc = 0L

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.GP_OUT_IN_KW_KH)
    }
    if (initBias == null) {
      biasInitMethod.init(bias, VariableFormat.ONE_D)
    }
    zeroGradParameters()
  }

  private def init(): Unit = {
    desc = FixPoint.FixConvOpCreate(0)
    ev.getType() match {
      case FloatType =>
        FixPoint.FixConvOpSetupConvParameter(desc, nOutputPlane, nInputPlane, nGroup,
          kernelH, kernelW, strideH, strideW, 1, 1, padH, padW,
          weight.storage().array().asInstanceOf[Array[Float]], weight.storageOffset() - 1,
          true, bias.storage().array().asInstanceOf[Array[Float]], bias.storageOffset() - 1, false)
      case _ => throw new UnsupportedOperationException(s"only support float")
    }
    FixPoint.FixConvOpQuantizeKernel(desc, 64.0f)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    if (!_init) {
      init()
      _init = true
    }

    val dimWidth = if (input.dim() == 3) 3 else 4
    val dimHeight = if (input.dim() == 3) 2 else 3
    val dimChannel = if (input.dim() == 3) 1 else 2
    val batchSize = if (input.dim() == 3)  1 else input.size(1)
    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)
    val outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1
    val outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1

    if (input.dim() == 3) {
      output.resize(Array(nOutputPlane, outputHeight, outputWidth))
    } else {
      output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))
    }

/*
    ev.getType() match {
      case FloatType =>
        FixPoint.FixConvOpSetupTargetBuffer(desc,
          output.storage().array().asInstanceOf[Array[Float]],
          output.storageOffset() - 1)
        FixPoint.FixConvOpQuantizeData(desc,
          batchSize, input.size(dimChannel), input.size(dimHeight), input.size(dimWidth),
          input.storage().array().asInstanceOf[Array[Float]], input.storageOffset() - 1, 127.0f)
      case _ => throw new UnsupportedOperationException(s"only support float")
    }
    FixPoint.FixConvOpExecute(desc, 0.5f)
*/
    ev.getType() match {
      case FloatType =>
        FixPoint.FixConvOpExecuteAll(desc, batchSize, input.size(dimChannel), input.size(dimHeight),
          input.size(dimWidth), input.storage().array().asInstanceOf[Array[Float]],
          input.storageOffset() - 1, 127.0f, output.storage().array().asInstanceOf[Array[Float]],
          output.storageOffset() - 1, 0.5f)
      case _ => throw new UnsupportedOperationException(s"only support float")
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
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "bias" -> bias,
      "gradWeight" -> gradWeight, "gradBias" -> gradBias))
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
      bias == other.bias &&
      gradWeight == other.gradWeight &&
      gradBias == other.gradBias
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
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()

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
    if (desc != 0) {
      FixPoint.FixConvOpFree(desc)
      desc = 0L
    }
  }
}
