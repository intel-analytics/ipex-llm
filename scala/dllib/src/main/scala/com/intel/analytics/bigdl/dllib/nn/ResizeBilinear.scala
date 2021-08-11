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

import com.intel.analytics.bigdl.nn.ResizeBilinear.InterpolationWeight
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Resize the input image with bilinear interpolation. The input image must be a float tensor with
 * NHWC or NCHW layout.
 *
 * @param outputHeight output height
 * @param outputWidth output width
 * @param alignCorners align corner or not
 * @param dataFormat the data format of the input image, NHWC or NCHW
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class ResizeBilinear[T: ClassTag](val outputHeight: Int, val outputWidth: Int,
  val alignCorners: Boolean,
  val dataFormat: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[Float], Tensor[Float], T]{

  private val ys = (1 to (outputHeight + 1)).map(i => InterpolationWeight(0, 0, 0)).toArray
  private val xs = (1 to (outputWidth + 1)).map(i => InterpolationWeight(0, 0, 0)).toArray

  import ResizeBilinear._

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    require(input.nDimension() == 4, "only accept 4D input")
    require(input.isContiguous(), "only accept contiguous input")

    dataFormat match {
      case DataFormat.NHWC => updateOutputNHWC(input)
      case DataFormat.NCHW => updateOutputNCHW(input)
    }

    output
  }

  private def updateOutputNCHW(input: Tensor[Float]): Tensor[Float] = {
    val batchSize = input.size(1)
    val channels = input.size(2)
    val inHeight = input.size(3)
    val inWidth = input.size(4)

    if (inHeight == outputHeight && inWidth == outputWidth) {
      output = input
      output
    } else {
      computeInterpolationWeights(outputHeight, inHeight,
        calculateResizeScale(inHeight, outputHeight, alignCorners), ys)
      computeInterpolationWeights(outputWidth, inWidth,
        calculateResizeScale(inWidth, outputWidth, alignCorners), xs)

      output.resize(batchSize, channels, outputHeight, outputWidth)

      var i = 0
      while (i < batchSize * channels) {
        val inputOffset = (input.storageOffset() - 1) + i * inHeight * inWidth
        val outputOffset = (output.storageOffset() - 1) + i * outputHeight * outputWidth

        if (input.getType() == FloatType) {
          resizeImage(input.storage().array(),
            inputOffset, 1, inHeight, inWidth,
            outputHeight, outputWidth, 1, xs, ys,
            output.storage().array(),
            outputOffset)
        } else if (input.getType() == DoubleType) {
          resizeImage(input.storage().array(),
            inputOffset, 1, inHeight, inWidth,
            outputHeight, outputWidth, 1, xs, ys,
            output.storage().array(),
            outputOffset)
        } else {
          throw new IllegalArgumentException(
            s"ResizeBilinear does not support type ${input.getType()}")
        }

        i += 1
      }

      output
    }
  }

  private def updateOutputNHWC(input: Tensor[Float]): Tensor[Float] = {
    val batchSize = input.size(1)
    val inHeight = input.size(2)
    val inWidth = input.size(3)
    val channels = input.size(4)

    if (inHeight == outputHeight && inWidth == outputWidth) {
      output = input
      output
    } else {
      computeInterpolationWeights(outputHeight, inHeight,
        calculateResizeScale(inHeight, outputHeight, alignCorners), ys)
      computeInterpolationWeights(outputWidth, inWidth,
        calculateResizeScale(inWidth, outputWidth, alignCorners), xs)

      var i = 0
      while(i < xs.size) {
        xs(i).lower *= channels
        xs(i).upper *= channels
        i += 1
      }

      output.resize(batchSize, outputHeight, outputWidth, channels)
      if (input.getType() == FloatType) {
        resizeImage(input.storage().array(),
          input.storageOffset() - 1, batchSize, inHeight, inWidth,
          outputHeight, outputWidth, channels, xs, ys,
          output.storage().array(),
          output.storageOffset() - 1)
      } else if (input.getType() == DoubleType) {
        resizeImage(input.storage().array(),
          input.storageOffset() - 1, batchSize, inHeight, inWidth,
          outputHeight, outputWidth, channels, xs, ys,
          output.storage().array(),
          output.storageOffset() - 1)
      } else {
        throw new IllegalArgumentException(
          s"ResizeBilinear does not support type ${input.getType()}")
      }

      output
    }
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    require(input.nDimension() == 4, "only accept 4D input")
    require(gradOutput.nDimension() == 4, "only accept 4D gradOutput")
    require(input.isContiguous(), "only accept contiguous input")
    require(gradOutput.isContiguous(), "only accept contiguous gradOutput")

    dataFormat match {
      case DataFormat.NHWC => updateGradInputNHWC(input, gradOutput)
      case DataFormat.NCHW => updateGradInputNCHW(input, gradOutput)
    }
    gradInput
  }

  private def updateGradInputNHWC(input: Tensor[Float],
                                  gradOutput: Tensor[Float]): Tensor[Float] = {
    val batchSize = input.size(1)
    val inHeight = input.size(2)
    val inWidth = input.size(3)
    val channels = input.size(4)

    require(gradOutput.size(2) == outputHeight, "output height is not match")
    require(gradOutput.size(3) == outputWidth, "output width is not match")

    val heightScale = calculateResizeScale(inHeight, outputHeight, alignCorners)
    val widthScale = calculateResizeScale(inWidth, outputWidth, alignCorners)

    gradInput.resizeAs(input)
    gradInput.zero()

    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1

    if (input.getType() == FloatType) {
      resizeImageBackprop(batchSize, channels, inHeight, inWidth,
        outputHeight, outputWidth, heightScale, widthScale,
        gradInputData, gradInputOffset,
        gradOutputData, gradOutputOffset)
    } else if (input.getType() == DoubleType) {
      resizeImageBackprop(batchSize, channels, inHeight, inWidth,
        outputHeight, outputWidth, heightScale, widthScale,
        gradInputData, gradInputOffset,
        gradOutputData, gradOutputOffset)
    } else {
      throw new IllegalArgumentException(
        s"ResizeBilinear does not support type ${input.getType()}")
    }

    gradInput
  }

  private def updateGradInputNCHW(input: Tensor[Float],
                                  gradOutput: Tensor[Float]): Tensor[Float] = {
    val batchSize = input.size(1)
    val channels = input.size(2)
    val inHeight = input.size(3)
    val inWidth = input.size(4)

    require(gradOutput.size(3) == outputHeight, "output height is not match")
    require(gradOutput.size(4) == outputWidth, "output width is not match")

    val heightScale = calculateResizeScale(inHeight, outputHeight, alignCorners)
    val widthScale = calculateResizeScale(inWidth, outputWidth, alignCorners)

    gradInput.resizeAs(input)
    gradInput.zero()

    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1

    var i = 0
    while (i < batchSize * channels) {
      val inOffset = gradInputOffset + i * (inHeight * inWidth)
      val outOffset = gradOutputOffset + i * (outputHeight * outputWidth)
      if (input.getType() == FloatType) {
        resizeImageBackprop(1, 1, inHeight, inWidth,
          outputHeight, outputWidth, heightScale, widthScale,
          gradInputData, inOffset,
          gradOutputData, outOffset)
      } else if (input.getType() == DoubleType) {
        resizeImageBackprop(1, 1, inHeight, inWidth,
          outputHeight, outputWidth, heightScale, widthScale,
          gradInputData, inOffset,
          gradOutputData, outOffset)
      } else {
        throw new IllegalArgumentException(
          s"ResizeBilinear does not support type ${input.getType()}")
      }
      i += 1
    }

    gradInput
  }
}

object ResizeBilinear {

  def apply[T: ClassTag](outputHeight: Int, outputWidth: Int, alignCorners: Boolean = false,
                         dataFormat: DataFormat = DataFormat.NCHW)
    (implicit ev: TensorNumeric[T]): ResizeBilinear[T] = {
    new ResizeBilinear[T](outputHeight, outputWidth, alignCorners, dataFormat)
  }

  private def computeLERP(
    topLeft: Float,
    topRight: Float,
    bottomLeft: Float,
    bottomRight: Float,
    xLERP: Float,
    yLERP: Float
  ): Float = {
    val top = topLeft + (topRight - topLeft) * xLERP
    val bottom = bottomLeft + (bottomRight - bottomLeft) * xLERP
    top + (bottom - top) * yLERP
  }

  private def computeInterpolationWeights(
    outSize: Int,
    inSize: Int,
    scale: Float,
    interpolation: Array[InterpolationWeight]
  ): Unit = {
    interpolation(outSize).lower = 0
    interpolation(outSize).upper = 0
    var i = outSize - 1
    while(i >= 0) {
      val in = i * scale
      interpolation(i).lower = in.toInt
      interpolation(i).upper = Math.min(interpolation(i).lower + 1, inSize - 1)
      interpolation(i).lerp = in - interpolation(i).lower
      i -= 1
    }
  }

  /**
   * Resize image
   * @param image NHWC input
   * @param batchSize
   * @param inHeight
   * @param inWidth
   * @param outHeight
   * @param outWidth
   * @param channels
   * @param xs
   * @param ys
   * @param output
   */
  @inline
  private def resizeImage(
    image: Array[Float], imageOffset: Int,
    batchSize: Int,
    inHeight: Int, inWidth: Int,
    outHeight: Int, outWidth: Int,
    channels: Int,
    xs: Array[InterpolationWeight],
    ys: Array[InterpolationWeight],
    output: Array[Float], outputOffset: Int
  ): Unit = {
    val inRowSize = inWidth * channels
    val inBatchNumber = inHeight * inRowSize
    val outRowSize = outWidth * channels
    var _imageOffset = imageOffset
    var _outputOffset = outputOffset

    // TODO: use multiple thread to speed up this
    var b = 0
    while(b < batchSize) {
      var y = 0
      while(y < outHeight) {
        val ysLERP = ys(y).lerp
        var x = 0
        while(x < outWidth) {
          val xsLower = xs(x).lower
          val xsUpper = xs(x).upper
          val xsLERP = xs(x).lerp

          var c = 0
          while(c < channels) {
            val topLeft = image(_imageOffset + ys(y).lower * inRowSize + xsLower + c)
            val topRight = image(_imageOffset + ys(y).lower * inRowSize + xsUpper + c)
            val bottomLeft = image(_imageOffset + ys(y).upper * inRowSize + xsLower + c)
            val bottomRight = image(_imageOffset + ys(y).upper * inRowSize + xsUpper + c)
            output(_outputOffset + x * channels + c) = computeLERP(topLeft, topRight, bottomLeft,
              bottomRight, xsLERP, ysLERP)
            c += 1
          }
          x += 1
        }
        _outputOffset += outRowSize
        y += 1
      }
      _imageOffset += inBatchNumber
      b += 1
    }
  }

  @inline
  private def resizeImageBackprop(
    batchSize: Int,
    channels: Int,
    inHeight: Int,
    inWidth: Int,
    outputHeight: Int,
    outputWidth: Int,
    heightScale: Float,
    widthScale: Float,
    gradInputData: Array[Float],
    gradInputOffset: Int,
    gradOutputData: Array[Float],
    gradOutputOffset: Int): Unit = {
    val inRowSize = inWidth * channels
    val inBatchNum = inHeight * inRowSize
    val outRowSize = outputWidth * channels
    val outBatchNum = outputHeight * outRowSize
    var b = 0
    while(b < batchSize) {
      var y = 0
      while(y < outputHeight) {
        val inY = y * heightScale
        val topY = inY.toInt
        val bottomY = math.min(math.ceil(inY).toInt, inHeight - 1)
        val yLERP = inY - topY
        val inverseYLERP = (1.0f - yLERP)
        var x = 0
        while(x < outputWidth) {
          val inX = x * widthScale
          val leftX = inX.toInt
          val rightX = math.min(math.ceil(inX).toInt, inWidth - 1)
          val xLERP = inX - leftX
          val inverseXLERP = (1.0f - xLERP)
          var c = 0
          while(c < channels) {
            gradInputData(gradInputOffset + b * inBatchNum + topY * inRowSize +
              leftX * channels + c) += gradOutputData(gradOutputOffset + b * outBatchNum +
              y * outRowSize + x * channels + c) * inverseYLERP * inverseXLERP
            gradInputData(gradInputOffset + b * inBatchNum + topY * inRowSize +
              rightX * channels + c) += gradOutputData(gradOutputOffset + b * outBatchNum +
              y * outRowSize + x * channels + c) * inverseYLERP * xLERP
            gradInputData(gradInputOffset + b * inBatchNum + bottomY * inRowSize +
              leftX * channels + c) += gradOutputData(gradOutputOffset + b * outBatchNum +
              y * outRowSize + x * channels + c) * yLERP * inverseXLERP
            gradInputData(gradInputOffset + b * inBatchNum + bottomY * inRowSize +
              rightX * channels + c) += gradOutputData(gradOutputOffset + b * outBatchNum +
              y * outRowSize + x * channels + c) * yLERP * xLERP
            c += 1
          }
          x += 1
        }
        y += 1
      }
      b += 1
    }
  }

  private case class InterpolationWeight(var lower: Int, var upper: Int, var lerp: Float)

  private def calculateResizeScale(inSize: Int, outSize: Int, alignCorners: Boolean): Float = {
    if (alignCorners && outSize > 1) {
      (inSize - 1).toFloat / (outSize - 1)
    } else {
      inSize.toFloat / outSize
    }
  }
}
