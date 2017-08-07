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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Applies 1D max-pooling operation in kW regions by step size dW steps.
 * Input sequence composed of nInputFrame frames.
 * The input tensor in forward(input) is expected to be a 2D tensor (nInputFrame x inputFrameSize)
 * or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).
 *
 * If the input sequence is a 2D tensor of dimension nInputFrame x inputFrameSize,
 * the output sequence will be nOutputFrame x inputFrameSize where
 *
 * nOutputFrame = (nInputFrame - kW) / dW + 1
 *
 * @param kW kernel width
 * @param dW step size in width
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class TemporalMaxPooling[T: ClassTag](
  val kW: Int, var dW: Int = -1)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  @transient var indices: Tensor[T] = _

  if (dW == -1) {
    dW = kW
  }

  def shapeCheck(
    input: Tensor[T],
    gradOutput: Tensor[T],
    indices: Tensor[T],
    kW: Int,
    dW: Int
  ): Unit = {
    var dimS = 1 // sequence dimension
    var dimF = 2 // feature dimension
    val ndims = input.nDimension()

    if (input.nDimension() == 3)
    {
      dimS = 2
      dimF = 3
    }

    val niframe = input.size(dimS)
    val framesize = input.size(dimF)
    val noframe = (niframe - kW) / dW + 1

    require(kW > 0,
      s"kernel size should be greater than zero, but got kW: $kW")
    require(dW > 0,
      s"stride should be greater than zero, but got dW: $dW")

    require(input.nDimension() == 2 || input.nDimension() == 3,
      s"2D or 3D (batch mode) tensor expected for input, but got: ${input.nDimension()}")
    require(input.size(dimS) >= kW,
      s"input sequence smaller than kernel size. Got: ${input.size(dimS)}, Expected: $kW")

    if (gradOutput != null) {
      require(gradOutput.nDimension() == ndims && gradOutput.size(dimS) == noframe)
      require(gradOutput.nDimension() == ndims && gradOutput.size(dimF) == framesize)
    }
    if (indices != null) {
      require(indices.nDimension() == ndims && indices.size(dimS) == noframe)
      require(indices.nDimension() == ndims && indices.size(dimF) == framesize)
    }
  }


  def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (indices == null) {
      indices = Tensor()
    }

    var nIFrame = 0
    var nOFrame = 0
    var frameSize = 0

    var dimS = 1 // sequence dimension
    var dimF = 2 // feature dimension

    shapeCheck(input, null, null, kW, dW)

    if (input.nDimension() == 3) {
      dimS = 2
      dimF = 3
    }

    nIFrame = input.size(dimS)
    frameSize = input.size(dimF)
    nOFrame = (nIFrame - kW) / dW + 1

    val inputC = input.contiguous()

    if (inputC.nDimension() == 2) {
      output.resize(nOFrame, frameSize)
      indices.resize(nOFrame, frameSize)

      ev.getType() match {
        case DoubleType => NNPrimitive.temporalMaxPoolingForwardDouble(
          inputC.asInstanceOf[Tensor[Double]].storage().array(), inputC.storageOffset() - 1,
          output.asInstanceOf[Tensor[Double]].storage().array(), output.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Double]].storage().array(), indices.storageOffset() - 1,
          nOFrame, frameSize, kW, dW
        )
        case FloatType => NNPrimitive.temporalMaxPoolingForwardFloat(
          inputC.asInstanceOf[Tensor[Float]].storage().array(), inputC.storageOffset() - 1,
          output.asInstanceOf[Tensor[Float]].storage().array(), output.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Float]].storage().array(), indices.storageOffset() - 1,
          nOFrame, frameSize, kW, dW
        )
        case _ => throw new UnsupportedOperationException(
          "TemporalMaxPooling: only Float/Double type supported")
      }
    } else {
      val nbFrame = input.size(1)
      output.resize(nbFrame, nOFrame, frameSize)
      indices.resize(nbFrame, nOFrame, frameSize)

      var i = 1
      while (i <= nbFrame) {
        val curInput = inputC(i)
        val curOutput = output(i)
        val curIndices = indices(i)

        ev.getType() match {
          case DoubleType => NNPrimitive.temporalMaxPoolingForwardDouble(
            curInput.asInstanceOf[Tensor[Double]].storage().array(),
            curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Double]].storage().array(),
            curIndices.storageOffset() - 1,
            nOFrame, frameSize, kW, dW
          )
          case FloatType => NNPrimitive.temporalMaxPoolingForwardFloat(
            curInput.asInstanceOf[Tensor[Float]].storage().array(),
            curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Float]].storage().array(),
            curOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Float]].storage().array(),
            curIndices.storageOffset() - 1,
            nOFrame, frameSize, kW, dW
          )
          case _ => throw new UnsupportedOperationException(
            "TemporalMaxPooling: only Float/Double type supported")
        }
        i += 1
      }
    }

    output
  }

  def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    shapeCheck(input, gradOutput, indices, kW, dW)

    val gradOutputC = gradOutput.contiguous()

    gradInput.resizeAs(input)
    gradInput.zero()

    var dimS = 1 // sequence dimension
    var dimF = 2 // feature dimension

    if (input.nDimension() == 3) {
      dimS = 2
      dimF = 3
    }

    val nOFrame = gradOutputC.size(dimS)
    val frameSize = gradOutputC.size(dimF)

    if (input.dim() == 2) {
      ev.getType() match {
        case DoubleType => NNPrimitive.temporalMaxPoolingBackwardDouble(
          gradInput.asInstanceOf[Tensor[Double]].storage().array(),
          gradInput.storageOffset() - 1,
          gradOutputC.asInstanceOf[Tensor[Double]].storage().array(),
          gradOutputC.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Double]].storage().array(),
          indices.storageOffset() - 1,
          nOFrame, frameSize, kW, dW
        )
        case FloatType => NNPrimitive.temporalMaxPoolingBackwardFloat(
          gradInput.asInstanceOf[Tensor[Float]].storage().array(),
          gradInput.storageOffset() - 1,
          gradOutputC.asInstanceOf[Tensor[Float]].storage().array(),
          gradOutputC.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Float]].storage().array(),
          indices.storageOffset() - 1,
          nOFrame, frameSize, kW, dW
        )
        case _ => throw new UnsupportedOperationException(
          "TemporalMaxPooling: only Float/Double type supported")
      }
    } else {
      val nBFrame = input.size(1)
      var i = 1
      while (i <= nBFrame) {
        val curGradInput = gradInput(i)
        val curGradOutput = gradOutputC(i)
        val curIndices = indices(i)

        ev.getType() match {
          case DoubleType => NNPrimitive.temporalMaxPoolingBackwardDouble(
            curGradInput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Double]].storage().array(),
            curIndices.storageOffset() - 1,
            nOFrame, frameSize, kW, dW
          )
          case FloatType => NNPrimitive.temporalMaxPoolingBackwardFloat(
            curGradInput.asInstanceOf[Tensor[Float]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Float]].storage().array(),
            curGradOutput.storageOffset() - 1,
            curIndices.asInstanceOf[Tensor[Float]].storage().array(),
            curIndices.storageOffset() - 1,
            nOFrame, frameSize, kW, dW
          )
          case _ => throw new UnsupportedOperationException(
            "TemporalMaxPooling: only Float/Double type supported")
        }
        i += 1
      }
    }

    gradInput
  }


  override def canEqual(other: Any): Boolean = other.isInstanceOf[TemporalMaxPooling[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TemporalMaxPooling[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        this.kW == that.kW &&
        this.dW == that.dW
    case _ => false
  }

  override def hashCode() : Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()

    val state = Seq(super.hashCode(), kW, dW)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def toString(): String = {
    s"${getPrintName()}($kW, $dW)"
  }

  override def clearState(): this.type = {
    super.clearState()
    indices.set()
    this
  }
}

object TemporalMaxPooling {
  def apply[@specialized(Float, Double) T: ClassTag](
    kW: Int,
    dW: Int = 1)(implicit ev: TensorNumeric[T]): TemporalMaxPooling[T] = {
    new TemporalMaxPooling[T](kW, dW)
  }
}
