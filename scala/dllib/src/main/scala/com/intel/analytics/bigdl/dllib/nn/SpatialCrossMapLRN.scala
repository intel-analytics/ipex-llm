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

import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.Future
import scala.reflect._

/**
 * Applies Spatial Local Response Normalization between different feature maps.
 * The operation implemented is:
 *                              x_f
 * y_f =  -------------------------------------------------
 *         (k+(alpha/size)* sum_{l=l1 to l2} (x_l^2^))^beta^
 *
 * where x_f is the input at spatial locations h,w (not shown for simplicity) and feature map f,
 * l1 corresponds to max(0,f-ceil(size/2)) and l2 to min(F, f-ceil(size/2) + size).
 * Here, F is the number of feature maps.
 * @param size  the number of channels to sum over (for cross channel LRN)
 * @param alpha  the scaling parameter
 * @param beta   the exponent
 * @param k
 */
@SerialVersionUID(3641570491004969703L)
class SpatialCrossMapLRN[T: ClassTag]
(val size: Int = 5, val alpha: Double = 1.0, val beta: Double = 0.75, val k: Double = 1.0,
  val format: DataFormat = DataFormat.NCHW)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient
  private var scale: Tensor[T] = null

  @transient
  private var paddedRatio: Tensor[T] = null

  @transient
  private var accumRatio: Tensor[T] = null

  @transient
  private var results: Array[Future[Unit]] = null

  require(size % 2 == 1, "LRN only supports odd values for size" +
    s"size $size")
  val prePad = (size - 1) / 2

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialCrossMapLRN[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialCrossMapLRN[T]]
    if (this.eq(other)) {
      return true
    }

    size == other.size &&
      alpha == other.alpha && beta == other.beta && k == other.k
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + size.hashCode()
    hash = hash * seed + alpha.hashCode()
    hash = hash * seed + beta.hashCode()
    hash = hash * seed + k.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($size, $alpha, $beta, $k)"
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "Input must have 4 dimensions, corresponding to " +
      "(batch, channels, height, width)" +
      s"input dimension ${input.nDimension()}")
    require(input.isContiguous(), "Input is not contiguous")

    output.resizeAs(input)
    if (scale == null) {
      scale = Tensor[T]().resizeAs(input)
    }
    scale.resizeAs(input)

    val batchNum = input.size(1)
    if (results == null || results.length != batchNum) {
      results = new Array[Future[Unit]](batchNum)
    }

    var b = 1
    while (b <= batchNum) {
      val _b = b
      results(b - 1) = Engine.model.invoke(() => {
        if (format == DataFormat.NCHW) {
          SpatialCrossMapLRN.forwardFrameNCHW(input.select(1, _b), output.select(1, _b),
            scale.select(1, _b), alpha, size, beta, k)
        } else {
          if (ev.getType() == FloatType) {
            SpatialCrossMapLRN.forwardFrameNHWCFloat(
              input.select(1, _b).asInstanceOf[Tensor[Float]],
              output.select(1, _b).asInstanceOf[Tensor[Float]],
              alpha, size, beta, k)
          } else {
            throw new NotImplementedError(s"Not support numeric type ${ev.getType()} in NHWC")
          }
        }
      })
      b += 1
    }
    Engine.model.sync(results)
    this.output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4, "Input must have 4 dimensions, corresponding to " +
      "(batch, channels, height, width)" +
      s"inputdimension ${input.nDimension()}")
    require(gradOutput.isContiguous(), "gradOutput is not contiguous")

    val batchNum = input.size(1)
    val channel = input.size(2)
    val height = input.size(3)
    val width = input.size(4)

    if (paddedRatio == null) {
      paddedRatio = Tensor[T]().resize(batchNum, channel + size - 1, height, width)
    }

    if (accumRatio == null) {
      accumRatio = Tensor[T]().resize(batchNum, height, width)
    }

    gradInput.resizeAs(input)

    if (results == null || results.length != batchNum) {
      results = new Array[Future[Unit]](batchNum)
    }

    var b = 1
    while (b <= batchNum) {
      val _b = b
      results(b - 1) = Engine.model.invoke(() => {
        if (format == DataFormat.NCHW) {
          SpatialCrossMapLRN.backwardFrameNCHW(input.select(1, _b), output.select(1, _b),
            scale.select(1, _b), gradOutput.select(1, _b), gradInput.select(1, _b),
            paddedRatio.select(1, _b), accumRatio.select(1, _b), alpha, size, beta)
        } else {
          if (ev.getType() == FloatType) {
            SpatialCrossMapLRN.backwardFrameNHWCFloat(
              gradOutput.select(1, _b).asInstanceOf[Tensor[Float]],
              input.select(1, _b).asInstanceOf[Tensor[Float]],
              gradInput.select(1, _b).asInstanceOf[Tensor[Float]],
              output.select(1, _b).asInstanceOf[Tensor[Float]],
              alpha, size, beta, k)
          } else {
            throw new NotImplementedError(s"Not support numeric type ${ev.getType()} in NHWC")
          }
        }
      })
      b += 1
    }
    Engine.model.sync(results)

    this.gradInput
  }
}

object SpatialCrossMapLRN {

  def apply[@specialized(Float, Double) T: ClassTag](
      size: Int = 5,
      alpha: Double = 1.0,
      beta: Double = 0.75,
      k: Double = 1.0,
      format: DataFormat = DataFormat.NCHW)
    (implicit ev: TensorNumeric[T]) : SpatialCrossMapLRN[T] = {
    new SpatialCrossMapLRN[T](size, alpha, beta, k, format)
  }

  private[bigdl] def forwardFrameNCHW[T](input: Tensor[T], output: Tensor[T],
    scale: Tensor[T], alpha: Double, size: Int, beta: Double, k: Double)
    (implicit ev: TensorNumeric[T]): Unit = {
    val channels = input.size(1)

    val inputSquare = output
    inputSquare.pow(input, ev.fromType(2))
    val prePad = (size - 1) / 2 + 1
    val prePadCrop = if (prePad > channels) channels else prePad
    val scaleFirst = scale.select(1, 1).zero()

    var c = 1
    while (c <= prePadCrop) {
      scaleFirst.add(inputSquare.select(1, c))
      c += 1
    }

    c = 2
    while (c <= channels) {
      val scalePrevious = scale.select(1, c - 1)
      val scaleCurrent = scale.select(1, c)
      scaleCurrent.copy(scalePrevious)
      if (c < channels - prePad + 2) {
        val squareNext = inputSquare.select(1, c + prePad - 1)
        scaleCurrent.add(ev.fromType(1), squareNext)
      }
      if (c > prePad) {
        val squarePrevious = inputSquare.select(1, c - prePad)
        scaleCurrent.add(ev.fromType(-1), squarePrevious)
      }
      c += 1
    }

    scale.mul(ev.fromType(alpha / size)).add(ev.fromType(k))
    output.pow(scale, ev.fromType(-beta))
    output.cmul(input)
  }

  def forwardFrameNHWCFloat(
    input: Tensor[Float],
    output: Tensor[Float],
    alpha: Double,
    size: Int,
    beta: Double,
    k: Double
  ): Unit = {
    require(input.isContiguous(), "input of LRN for NHWC should be contiguous")
    require(output.isContiguous(), "output of LRN for NHWC should be contiguous")
    val channel = input.size(3)
    val inputOffset = input.storageOffset() - 1
    val inputArray = input.storage().array()
    val outputOffset = output.storageOffset() - 1
    val outputArray = output.storage().array()
    val nElement = output.nElement()
    var l2sum = 0f
    var i = 0
    while(i < nElement) {
      val p = i % channel
      if (p == 0) {
        var c = 0
        l2sum = 0
        val depth = Math.min((size - 1) / 2 + 1, channel)
        while (c < depth) {
          val x = inputArray(inputOffset + i + c)
          l2sum += x * x
          c += 1
        }
      } else {
        if (p + (size - 1) / 2 < channel) {
          val x = inputArray(inputOffset + i + (size - 1) / 2)
          l2sum += x * x
        }
        if (p - (size - 1) / 2 > 0) {
          val x = inputArray(inputOffset + i - (size - 1) / 2 - 1)
          l2sum -= x * x
        }
      }
      outputArray(outputOffset + i) = inputArray(inputOffset + i) *
        Math.pow(k + alpha / size * l2sum, -beta).toFloat
      i += 1
    }
  }

  private def backwardFrameNCHW[T](
    input: Tensor[T], output: Tensor[T], scale: Tensor[T],
    gradOutput: Tensor[T], gradInput: Tensor[T], paddedRatio: Tensor[T],
    accumRatio: Tensor[T], alpha: Double, size: Int, beta: Double)
    (implicit ev: TensorNumeric[T]): Unit = {

    val channels = input.size(1)
    val inversePrePad = size - (size - 1) / 2
    val cacheRatioValue = ev.fromType(-2 * alpha * beta / size)

    gradInput.pow(scale, ev.fromType(-beta)).cmul(gradOutput)
    paddedRatio.zero()
    val paddedRatioCenter = paddedRatio.narrow(1, inversePrePad, channels)
    paddedRatioCenter.cmul(gradOutput, output).cdiv(scale)
    accumRatio.sum(paddedRatio.narrow(1, 1, size - 1), 1)
    var c = 1
    while (c <= channels) {
      accumRatio.add(paddedRatio.select(1, c + size - 1))
      gradInput.select(1, c).addcmul(cacheRatioValue, input.select(1, c), accumRatio)
      accumRatio.add(ev.fromType(-1), paddedRatio.select(1, c))
      c += 1
    }
  }

  private[bigdl] def backwardFrameNHWCFloat(
    gradOutput: Tensor[Float],
    input: Tensor[Float],
    gradInput: Tensor[Float],
    output: Tensor[Float],
    alpha: Double,
    size: Int,
    beta: Double,
    k: Double
  ): Unit = {
    gradInput.copy(input)
    val channel = input.size(3)
    val inputOffset = input.storageOffset() - 1
    val inputArray = input.storage().array()
    val outputOffset = output.storageOffset() - 1
    val outputArray = output.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradOutputArray = gradOutput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradInputArray = gradInput.storage().array()
    val nElement = gradInput.nElement()
    var glsum = 0f
    var i = 0
    while(i < nElement) {
      val p = i % channel
      if (p == 0) {
        var c = 0
        glsum = 0
        val depth = Math.min((size - 1) / 2 + 1, channel)
        while (c < depth) {
          val x = inputArray(inputOffset + i + c)
          val g = gradOutputArray(gradOutputOffset + i + c)
          val o = outputArray(outputOffset + i + c)
          glsum += g * Math.pow(o / x, (beta + 1) / beta).toFloat * x
          c += 1
        }
      } else {
        if (p + (size - 1) / 2 < channel) {
          val x = inputArray(inputOffset + i + (size - 1) / 2)
          val g = gradOutputArray(gradOutputOffset + i + (size - 1) / 2)
          val o = outputArray(outputOffset + i + (size - 1) / 2)
          glsum += g * Math.pow(o / x, (beta + 1) / beta).toFloat * x
        }
        if (p - (size - 1) / 2 - 1>= 0) {
          val x = inputArray(inputOffset + i - (size - 1) / 2 - 1)
          val g = gradOutputArray(gradOutputOffset + i - (size - 1) / 2 - 1)
          val o = outputArray(outputOffset + i - (size - 1) / 2 - 1)
          glsum -= g * Math.pow(o / x, (beta + 1) / beta).toFloat * x
        }
      }
      val x = inputArray(inputOffset + i)
      val g = gradOutputArray(gradOutputOffset + i)
      val o = outputArray(outputOffset + i)
      gradInputArray(gradInputOffset + i) =
        (o / x * g - 2 * beta * alpha / size * x * glsum).toFloat

      i += 1
    }
  }
}
