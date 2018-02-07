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

import java.util

import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect._
import com.intel.analytics.bigdl.utils.Engine

/**
 * Applies 2D average-pooling operation in kWxkH regions by step size dWxdH steps.
 * The number of output features is equal to the number of input planes.
 *
 * When padW and padH are both -1, we use a padding algorithm similar to the "SAME"
 * padding of tensorflow. That is
 *
 * outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
 * outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)
 *
 * padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
 * padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)
 *
 * padTop = padAlongHeight / 2
 * padLeft = padAlongWidth / 2
 *
 * @param kW kernel width
 * @param kH kernel height
 * @param dW step width
 * @param dH step height
 * @param padW padding width
 * @param padH padding height
 * @param globalPooling If globalPooling then it will pool over the size of the input by doing
 *                      kH = input->height and kW = input->width
 * @param ceilMode whether the output size is to be ceiled or floored
 * @param countIncludePad whether to include padding when dividing the
 *                        number of elements in pooling region
 * @param divide whether to do the averaging
 * @param format          DataFormat.NCHW or DataFormat.NHWC, indicating the input
 *                        data format
 */
@SerialVersionUID(4533142511857387857L)
class SpatialAveragePooling[T: ClassTag](
  var kW: Int,
  var kH: Int,
  val dW: Int = 1,
  val dH: Int = 1,
  val padW: Int = 0,
  val padH: Int = 0,
  val globalPooling: Boolean = false,
  var ceilMode: Boolean = false,
  private var countIncludePad: Boolean = true,
  private var divide: Boolean = true,
  val format: DataFormat = DataFormat.NCHW
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient
  private var results: Array[Future[Unit]] = null

  /**
   * set ceil mode
   * @return this
   */
  def ceil(): SpatialAveragePooling[T] = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): SpatialAveragePooling[T] = {
    ceilMode = false
    this
  }

  /**
   * set countIncludePad to true
   * @return this
   */
  def setCountIncludePad(): SpatialAveragePooling[T] = {
    countIncludePad = true
    this
  }

  /**
   * set countIncludePad to false
   * @return this
   */
  def setCountExcludePad(): SpatialAveragePooling[T] = {
    countIncludePad = false
    this
  }

  private def updateOutputFrameDouble(input: Tensor[Double], output: Tensor[Double],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int,
    outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(input.isContiguous())
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    var k = 0
    while (k < nInputPlane) {
      var yy = 0
      while (yy < outputHeight) {
        var xx = 0
        while (xx < outputWidth) {
          var hStart = yy * dH - padTop
          var wStart = xx * dW - padLeft
          var hEnd = math.min(hStart + kH, inputHeight + padBottom)
          var wEnd = math.min(wStart + kW, inputWidth + padRight)
          val poolSize = (hEnd - hStart) * (wEnd - wStart)
          hStart = math.max(hStart, 0)
          wStart = math.max(wStart, 0)
          hEnd = math.min(hEnd, inputHeight)
          wEnd = math.min(wEnd, inputWidth)

          var sum = 0.0
          val divideFactor = if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)
          var ky = hStart
          while (ky < hEnd) {
            var kx = wStart
            while (kx < wEnd) {
              sum += inputData(inputOffset + k * inputHeight * inputWidth + ky * inputWidth + kx)
              kx += 1
            }
            ky += 1
          }
          outputData(outputOffset + k * outputHeight * outputWidth + yy * outputWidth + xx) =
            sum / divideFactor
          xx += 1
        }
        yy += 1
      }
      k += 1
    }
  }

  private def updateOutputFrameFloat(input: Tensor[Float], output: Tensor[Float],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int, outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(input.isContiguous())
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    var k = 0
    while (k < nInputPlane) {
      var yy = 0
      while (yy < outputHeight) {
        var xx = 0
        while (xx < outputWidth) {
          var hStart = yy * dH - padTop
          var wStart = xx * dW - padLeft
          var hEnd = math.min(hStart + kH, inputHeight + padBottom)
          var wEnd = math.min(wStart + kW, inputWidth + padRight)
          val poolSize = (hEnd - hStart) * (wEnd - wStart)
          hStart = math.max(hStart, 0)
          wStart = math.max(wStart, 0)
          hEnd = math.min(hEnd, inputHeight)
          wEnd = math.min(wEnd, inputWidth)

          var sum = 0.0f
          val divideFactor = if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)
          var ky = hStart
          while (ky < hEnd) {
            var kx = wStart
            while (kx < wEnd) {
              sum += inputData(inputOffset + k * inputHeight * inputWidth + ky * inputWidth + kx)
              kx += 1
            }
            ky += 1
          }
          outputData(outputOffset + k * outputHeight * outputWidth + yy * outputWidth + xx) =
            sum / divideFactor
          xx += 1
        }
        yy += 1
      }
      k += 1
    }
  }

  private def updateOutputFrameDoubleNHWC(
     input: Tensor[Double], output: Tensor[Double],
     nInputPlane: Int, inputHeight: Int, inputWidth: Int, outputHeight: Int,
     outputWidth: Int, kW: Int, kH: Int, dW: Int, dH: Int,
     padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(input.isContiguous())
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    var yy = 0
    while (yy < outputHeight) {
      var xx = 0
      while (xx < outputWidth) {
        var hStart = yy * dH - padTop
        var wStart = xx * dW - padLeft
        var hEnd = math.min(hStart + kH, inputHeight + padBottom)
        var wEnd = math.min(wStart + kW, inputWidth + padRight)
        val poolSize = (hEnd - hStart) * (wEnd - wStart)
        hStart = math.max(hStart, 0)
        wStart = math.max(wStart, 0)
        hEnd = math.min(hEnd, inputHeight)
        wEnd = math.min(wEnd, inputWidth)

        val currOutLocStart = outputOffset + (yy * outputWidth + xx) * nInputPlane
        val currOutLocEnd = currOutLocStart + nInputPlane
        util.Arrays.fill(outputData, currOutLocStart, currOutLocEnd, 0.0)
        val divideFactor = if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)

        var y = hStart
        while (y < hEnd) {
          var x = wStart
          while (x < wEnd) {
            // k, y, x input indexers
            val tcntr = y *inputWidth + x
            val currInLocStart = inputOffset + tcntr * nInputPlane
            var n = 0
            while (n < nInputPlane) {
              val value = inputData(currInLocStart + n)
              outputData(currOutLocStart + n) += value
              n = n + 1
            }
            x += 1
          }
          y += 1
        }
        var n = 0
        while (n < nInputPlane) {
          outputData(currOutLocStart + n) /= divideFactor
          n = n + 1
        }
        xx += 1
      }
      yy += 1
    }
  }


  private def updateOutputFrameFloatNHWC(
    input: Tensor[Float], output: Tensor[Float],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int, outputHeight: Int,
    outputWidth: Int, kW: Int, kH: Int, dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(input.isContiguous())
    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset() - 1
    var yy = 0
    while (yy < outputHeight) {
      var xx = 0
      while (xx < outputWidth) {
        var hStart = yy * dH - padTop
        var wStart = xx * dW - padLeft
        var hEnd = math.min(hStart + kH, inputHeight + padBottom)
        var wEnd = math.min(wStart + kW, inputWidth + padRight)
        val poolSize = (hEnd - hStart) * (wEnd - wStart)
        hStart = math.max(hStart, 0)
        wStart = math.max(wStart, 0)
        hEnd = math.min(hEnd, inputHeight)
        wEnd = math.min(wEnd, inputWidth)

        val currOutLocStart = outputOffset + (yy * outputWidth + xx) * nInputPlane
        val currOutLocEnd = currOutLocStart + nInputPlane
        util.Arrays.fill(outputData, currOutLocStart, currOutLocEnd, 0.0f)
        val divideFactor = if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)

        var y = hStart
        while (y < hEnd) {
          var x = wStart
          while (x < wEnd) {
            // k, y, x input indexers
            val tcntr = y *inputWidth + x
            val currInLocStart = inputOffset + tcntr * nInputPlane
            var n = 0
            while (n < nInputPlane) {
              val value = inputData(currInLocStart + n)
              outputData(currOutLocStart + n) += value
              n = n + 1
            }
            x += 1
          }
          y += 1
        }
        var n = 0
        while (n < nInputPlane) {
          outputData(currOutLocStart + n) /= divideFactor
          n = n + 1
        }
        xx += 1
      }
      yy += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialAveragePooling: " + ErrorInfo.constrainInputAs3DOrBatch +
    s"input dimension ${input.dim()}")

    val (dimH, dimW, dimC) = format.getHWCDims(input.dim())

    val inputHeight = input.size(dimH)
    val inputWidth = input.size(dimW)
    val nInputPlane = input.size(dimC)
    if (globalPooling) {
      kH = inputHeight
      kW = inputWidth
    }

    val sizes =
      if (padW == -1 && padH == -1) {
        Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW)
      } else {
        Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode)
      }
    val padTop = sizes(0)
    val padBottom = sizes(1)
    val padLeft = sizes(2)
    val padRight = sizes(3)
    val outputHeight = sizes(4)
    val outputWidth = sizes(5)

    if (ceilMode && padW == 0 && (inputWidth - kW) % dW == 0) {
      ceilMode = false // The ceil mode is not needed.
    }

    if (input.dim() == 3) {
      format match {
        case DataFormat.NCHW =>
          output.resize(Array(nInputPlane, outputHeight, outputWidth))
          if (classTag[T] == classTag[Double]) {
            updateOutputFrameDouble(input.asInstanceOf[Tensor[Double]],
              output.asInstanceOf[Tensor[Double]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          } else {
            updateOutputFrameFloat(input.asInstanceOf[Tensor[Float]],
              output.asInstanceOf[Tensor[Float]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          }
        case DataFormat.NHWC =>
          output.resize(Array(outputHeight, outputWidth, nInputPlane))
          if (classTag[T] == classTag[Double]) {
            updateOutputFrameDoubleNHWC(input.asInstanceOf[Tensor[Double]],
              output.asInstanceOf[Tensor[Double]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          } else {
            updateOutputFrameFloatNHWC(input.asInstanceOf[Tensor[Float]],
              output.asInstanceOf[Tensor[Float]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          }
      }

    }
    else {
      val nbatch = input.size(1)
      if (results == null || results.length != nbatch) {
        results = new Array[Future[Unit]](nbatch)
      }
      format match {
        case DataFormat.NCHW =>
          output.resize(Array(nbatch, nInputPlane, outputHeight, outputWidth))

          var i = 1
          while (i <= nbatch) {
            val _i = i
            results(_i - 1) = Engine.model.invoke(() => {
              if (classTag[T] == classTag[Double]) {
                updateOutputFrameDouble(input(_i).asInstanceOf[Tensor[Double]],
                  output(_i).asInstanceOf[Tensor[Double]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              } else {
                updateOutputFrameFloat(input(_i).asInstanceOf[Tensor[Float]],
                  output(_i).asInstanceOf[Tensor[Float]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              }
            })
            i += 1
          }
          Engine.model.sync(results)
        case DataFormat.NHWC =>
          output.resize(Array(nbatch, outputHeight, outputWidth, nInputPlane))

          var i = 1
          while (i <= nbatch) {
            val _i = i
            results(_i - 1) = Engine.model.invoke(() => {
              if (classTag[T] == classTag[Double]) {
                updateOutputFrameDoubleNHWC(input(_i).asInstanceOf[Tensor[Double]],
                  output(_i).asInstanceOf[Tensor[Double]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              } else {
                updateOutputFrameFloatNHWC(input(_i).asInstanceOf[Tensor[Float]],
                  output(_i).asInstanceOf[Tensor[Float]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              }
            })
            i += 1
          }
          Engine.model.sync(results)
      }

    }

    if (!divide) {
      output.mul(ev.fromType[Int](kW * kH))
    }
    output
  }

  private def updateGradInputFrameDouble(gradInput: Tensor[Double], gradOutput: Tensor[Double],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int,
    outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(gradOutput.isContiguous())
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    var k = 0
    while (k < nInputPlane) {
      var yy = 0
      while (yy < outputHeight) {
        var xx = 0
        while (xx < outputWidth) {
          var hStart = yy * dH - padTop
          var wStart = xx * dW - padLeft
          var hEnd = math.min(hStart + kH, inputHeight + padBottom)
          var wEnd = math.min(wStart + kW, inputWidth + padRight)
          val poolSize = (hEnd - hStart) * (wEnd - wStart)
          hStart = math.max(hStart, 0)
          wStart = math.max(wStart, 0)
          hEnd = math.min(hEnd, inputHeight)
          wEnd = math.min(wEnd, inputWidth)
          val divideFactor = if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)
          val z =
            gradOutputData(gradOutputOffset + k * outputHeight * outputWidth +
              yy * outputWidth + xx)
          var ky = hStart
          while (ky < hEnd) {
            var kx = wStart
            while (kx < wEnd) {
              gradInputData(gradInputOffset +
                k * inputHeight * inputWidth + ky * inputWidth + kx) += z / divideFactor
              kx += 1
            }
            ky += 1
          }
          xx += 1
        }
        yy += 1
      }
      k += 1
    }
  }

  private def updateGradInputFrameFloat(gradInput: Tensor[Float], gradOutput: Tensor[Float],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int,
    outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(gradOutput.isContiguous())
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1
    var k = 0
    while (k < nInputPlane) {
      var yy = 0
      while (yy < outputHeight) {
        var xx = 0
        while (xx < outputWidth) {
          var hStart = yy * dH - padTop
          var wStart = xx * dW - padLeft
          var hEnd = math.min(hStart + kH, inputHeight + padBottom)
          var wEnd = math.min(wStart + kW, inputWidth + padRight)
          val poolSize = (hEnd - hStart) * (wEnd - wStart)
          hStart = math.max(hStart, 0)
          wStart = math.max(wStart, 0)
          hEnd = math.min(hEnd, inputHeight)
          wEnd = math.min(wEnd, inputWidth)
          val divideFactor =
            if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)
          val z = gradOutputData(gradOutputOffset + k * outputHeight * outputWidth +
            yy * outputWidth + xx)
          var ky = hStart
          while (ky < hEnd) {
            var kx = wStart
            while (kx < wEnd) {
              gradInputData(gradInputOffset + k * inputHeight * inputWidth + ky *
                inputWidth + kx) += z / divideFactor
              kx += 1
            }
            ky += 1
          }
          xx += 1
        }
        yy += 1
      }
      k += 1
    }
  }

  private def updateGradInputFrameDoubleNHWC(
    gradInput: Tensor[Double], gradOutput: Tensor[Double],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int,
    outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(gradOutput.isContiguous())
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1

    var yy = 0
    while (yy < outputHeight) {
      var xx = 0
      while (xx < outputWidth) {
        var hStart = yy * dH - padTop
        var wStart = xx * dW - padLeft
        var hEnd = math.min(hStart + kH, inputHeight + padBottom)
        var wEnd = math.min(wStart + kW, inputWidth + padRight)
        val poolSize = (hEnd - hStart) * (wEnd - wStart)
        hStart = math.max(hStart, 0)
        wStart = math.max(wStart, 0)
        hEnd = math.min(hEnd, inputHeight)
        wEnd = math.min(wEnd, inputWidth)

        val divideFactor =
          if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)

        val outputOffset = gradOutputOffset + (yy * outputWidth + xx) * nInputPlane
        var ky = hStart
        while (ky < hEnd) {
          var kx = wStart
          while (kx < wEnd) {
            var n = 0
            val inputOffset = gradInputOffset + (ky * inputWidth + kx) * nInputPlane
            while (n < nInputPlane) {
              gradInputData(inputOffset + n) += gradOutputData(outputOffset + n) / divideFactor
              n = n + 1
            }
            kx += 1
          }
          ky += 1
        }
        xx += 1
      }
      yy += 1
    }
  }

  private def updateGradInputFrameFloatNHWC(
     gradInput: Tensor[Float], gradOutput: Tensor[Float],
     nInputPlane: Int, inputHeight: Int, inputWidth: Int,
     outputHeight: Int, outputWidth: Int,
     kW: Int, kH: Int, dW: Int, dH: Int,
     padLeft: Int, padTop: Int, padRight: Int, padBottom: Int): Unit = {
    require(gradOutput.isContiguous())
    val gradInputData = gradInput.storage().array()
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradOutputData = gradOutput.storage().array()
    val gradOutputOffset = gradOutput.storageOffset() - 1

    var yy = 0
    while (yy < outputHeight) {
      var xx = 0
      while (xx < outputWidth) {
        var hStart = yy * dH - padTop
        var wStart = xx * dW - padLeft
        var hEnd = math.min(hStart + kH, inputHeight + padBottom)
        var wEnd = math.min(wStart + kW, inputWidth + padRight)
        val poolSize = (hEnd - hStart) * (wEnd - wStart)
        hStart = math.max(hStart, 0)
        wStart = math.max(wStart, 0)
        hEnd = math.min(hEnd, inputHeight)
        wEnd = math.min(wEnd, inputWidth)

        val divideFactor =
          if (countIncludePad) poolSize else (hEnd - hStart) * (wEnd - wStart)

        val outputOffset = gradOutputOffset + (yy * outputWidth + xx) * nInputPlane
        var ky = hStart
        while (ky < hEnd) {
          var kx = wStart
          while (kx < wEnd) {
            var n = 0
            val inputOffset = gradInputOffset + (ky * inputWidth + kx) * nInputPlane
            while (n < nInputPlane) {
              gradInputData(inputOffset + n) += gradOutputData(outputOffset + n) / divideFactor
              n = n + 1
            }
            kx += 1
          }
          ky += 1
        }
        xx += 1
      }
      yy += 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val inputSize = input.size()
    updateGradInputInternal(inputSize, gradOutput)
  }

  private[bigdl] def updateGradInputInternal(inputSize: Array[Int],
                                             gradOutput: Tensor[T]): Tensor[T] = {
    require(inputSize.length == 3 || inputSize.length == 4,
      "SpatialAveragePooling: " + ErrorInfo.constrainInputAs3DOrBatch +
    s"input dimension ${inputSize.length}")
    // dimh, dimw, dimc start with 1
    val (dimh, dimw, dimc) = format.getHWCDims(inputSize.length)

    val nInputPlane = inputSize(dimc - 1)
    val inputHeight = inputSize(dimh - 1)
    val inputWidth = inputSize(dimw - 1)

    val sizes =
      if (padW == -1 && padH == -1) {
        // no ceil/floor mode in SAME padding
        Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW)
      } else {
        require(inputWidth >= kW - padW && inputHeight >= kH - padH,
          "input smaller than kernel size")
        require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel size")
        Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode)
      }
    val padTop = sizes(0)
    val padBottom = sizes(1)
    val padLeft = sizes(2)
    val padRight = sizes(3)
    val outputHeight = sizes(4)
    val outputWidth = sizes(5)

    gradInput.resize(inputSize).zero()
    if (inputSize.length == 3) {
      format match {
        case DataFormat.NCHW =>
          if (classTag[T] == classTag[Double]) {
            updateGradInputFrameDouble(gradInput.asInstanceOf[Tensor[Double]],
              gradOutput.asInstanceOf[Tensor[Double]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          } else {
            updateGradInputFrameFloat(gradInput.asInstanceOf[Tensor[Float]],
              gradOutput.asInstanceOf[Tensor[Float]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          }
        case DataFormat.NHWC =>
          if (classTag[T] == classTag[Double]) {
            updateGradInputFrameDoubleNHWC(gradInput.asInstanceOf[Tensor[Double]],
              gradOutput.asInstanceOf[Tensor[Double]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          } else {
            updateGradInputFrameFloatNHWC(gradInput.asInstanceOf[Tensor[Float]],
              gradOutput.asInstanceOf[Tensor[Float]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
          }
      }
    } else {
      val nBatch = inputSize(0)

      if (results == null || results.length != nBatch) {
        results = new Array[Future[Unit]](nBatch)
      }

      format match {
        case DataFormat.NCHW =>
          var i = 1
          while (i <= nBatch) {
            val _i = i
            results(_i - 1) = Engine.model.invoke(() => {
              if (classTag[T] == classTag[Double]) {
                updateGradInputFrameDouble(gradInput(_i).asInstanceOf[Tensor[Double]],
                  gradOutput(_i).asInstanceOf[Tensor[Double]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              } else {
                updateGradInputFrameFloat(gradInput(_i).asInstanceOf[Tensor[Float]],
                  gradOutput(_i).asInstanceOf[Tensor[Float]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              }
            })
            i += 1
          }
          Engine.model.sync(results)
        case DataFormat.NHWC =>
          var i = 1
          while (i <= nBatch) {
            val _i = i
            results(_i - 1) = Engine.model.invoke(() => {
              if (classTag[T] == classTag[Double]) {
                updateGradInputFrameDoubleNHWC(gradInput(_i).asInstanceOf[Tensor[Double]],
                  gradOutput(_i).asInstanceOf[Tensor[Double]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              } else {
                updateGradInputFrameFloatNHWC(gradInput(_i).asInstanceOf[Tensor[Float]],
                  gradOutput(_i).asInstanceOf[Tensor[Float]],
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kW, kH, dW, dH, padLeft, padTop, padRight, padBottom)
              }
            })
            i += 1
          }
          Engine.model.sync(results)
      }


    }
    if (!divide) {
      gradInput.mul(ev.fromType[Int](kW * kH))
    }
    gradInput
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialAveragePooling[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialAveragePooling[T]]
    if (this.eq(other)) {
      return true
    }

    kW == other.kW && kH == other.kH && dW == other.dW && dH == other.dH && padW == other.padW &&
      padH == other.padH && ceilMode == other.ceilMode &&
      countIncludePad == other.countIncludePad && divide == other.divide
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + ceilMode.hashCode()
    hash = hash * seed + countIncludePad.hashCode()
    hash = hash * seed + divide.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($kW, $kH, $dW, $dH, $padW, $padH)"
  }
}

object SpatialAveragePooling {
  def apply[T: ClassTag](
      kW: Int,
      kH: Int,
      dW: Int = 1,
      dH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      globalPooling: Boolean = false,
      ceilMode: Boolean = false,
      countIncludePad: Boolean = true,
      divide: Boolean = true,
      format: DataFormat = DataFormat.NCHW)
      (implicit ev: TensorNumeric[T]) : SpatialAveragePooling[T] = {
    new SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH, globalPooling,
      ceilMode, countIncludePad, divide, format)
  }
}
