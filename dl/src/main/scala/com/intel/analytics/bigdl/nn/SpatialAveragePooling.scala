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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect._
import com.intel.analytics.bigdl.utils.Engine

@SerialVersionUID(4533142511857387857L)
class SpatialAveragePooling[@specialized(Float, Double) T: ClassTag](
  val kW: Int,
  val kH: Int,
  val dW: Int = 1,
  val dH: Int = 1,
  private var padW: Int = 0,
  private var padH: Int = 0,
  private var ceilMode: Boolean = false,
  private var countIncludePad: Boolean = true,
  private var divide: Boolean = true
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient
  private var results: Array[Future[Unit]] = null

  def ceil(): SpatialAveragePooling[T] = {
    ceilMode = true
    this
  }

  def floor(): SpatialAveragePooling[T] = {
    ceilMode = false
    this
  }

  def setCountIncludePad(): SpatialAveragePooling[T] = {
    countIncludePad = true
    this
  }

  def setCountExcludePad(): SpatialAveragePooling[T] = {
    countIncludePad = false
    this
  }

  private def updateOutputFrameDouble(input: Tensor[Double], output: Tensor[Double],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int,
    outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int): Unit = {
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
          var hStart = yy * dH - padH
          var wStart = xx * dW - padW
          var hEnd = math.min(hStart + kH, inputHeight + padH)
          var wEnd = math.min(wStart + kW, inputWidth + padW)
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
    kW: Int, kH: Int, dW: Int, dH: Int): Unit = {
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
          var hStart = yy * dH - padH
          var wStart = xx * dW - padW
          var hEnd = math.min(hStart + kH, inputHeight + padH)
          var wEnd = math.min(wStart + kW, inputWidth + padW)
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

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialAveragePooling: " + ErrorInfo.constrainInputAs3DOrBatch)
    val dimH = input.dim() - 1
    val dimW = input.dim()
    val inputHeight = input.size(dimH)
    val inputWidth = input.size(dimW)
    val nInputPlane = input.size(dimH - 1)
    var outputHeight =
      if (ceilMode) {
        math.ceil((inputHeight - kH + 2 * padH).toFloat / dH).toInt + 1
      } else {
        math.floor((inputHeight - kH + 2 * padH).toFloat / dH).toInt + 1
      }
    var outputWidth =
      if (ceilMode) {
        math.ceil((inputWidth - kW + 2 * padW).toFloat / dW).toInt + 1
      } else {
        math.floor((inputWidth - kW + 2 * padW).toFloat / dW).toInt + 1
      }
    if (padW != 0 || padH != 0) {
      // ensure that the last pooling starts inside the image
      // needed to avoid problems in ceil mode
      if ((outputHeight - 1) * dH >= inputHeight + padH) {
        outputHeight -= 1
      }
      if ((outputWidth - 1) * dW >= inputWidth + padW) {
        outputWidth -= 1
      }
    }
    if (input.dim() == 3) {
      output.resize(Array(nInputPlane, outputHeight, outputWidth))
      if (classTag[T] == classTag[Double]) {
        updateOutputFrameDouble(input.asInstanceOf[Tensor[Double]],
          output.asInstanceOf[Tensor[Double]],
          nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kW, kH, dW, dH)
      } else {
        updateOutputFrameFloat(input.asInstanceOf[Tensor[Float]],
          output.asInstanceOf[Tensor[Float]],
          nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kW, kH, dW, dH)
      }
    }
    else {
      val nbatch = input.size(1)
      output.resize(Array(nbatch, nInputPlane, outputHeight, outputWidth))

      if (results == null || results.length != nbatch) {
        results = new Array[Future[Unit]](nbatch)
      }

      var i = 1
      while (i <= nbatch) {
        val _i = i
        results(_i - 1) = Engine.model.invoke(() => {
          if (classTag[T] == classTag[Double]) {
            updateOutputFrameDouble(input(_i).asInstanceOf[Tensor[Double]],
              output(_i).asInstanceOf[Tensor[Double]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH)
          } else {
            updateOutputFrameFloat(input(_i).asInstanceOf[Tensor[Float]],
              output(_i).asInstanceOf[Tensor[Float]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH)
          }
        })
        i += 1
      }
      Engine.model.sync(results)
    }

    if (!divide) {
      output.mul(ev.fromType[Int](kW * kH))
    }
    output
  }

  private def updateGradInputFrameDouble(gradInput: Tensor[Double], gradOutput: Tensor[Double],
    nInputPlane: Int, inputHeight: Int, inputWidth: Int,
    outputHeight: Int, outputWidth: Int,
    kW: Int, kH: Int, dW: Int, dH: Int): Unit = {
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
          var hStart = yy * dH - padH
          var wStart = xx * dW - padW
          var hEnd = math.min(hStart + kH, inputHeight + padH)
          var wEnd = math.min(wStart + kW, inputWidth + padW)
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
    kW: Int, kH: Int, dW: Int, dH: Int): Unit = {
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
          var hStart = yy * dH - padH
          var wStart = xx * dW - padW
          var hEnd = math.min(hStart + kH, inputHeight + padH)
          var wEnd = math.min(wStart + kW, inputWidth + padW)
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

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialAveragePooling: " + ErrorInfo.constrainInputAs3DOrBatch)
    val dimh = input.dim() - 1
    val dimw = input.dim()
    val inputHeight = input.size(dimh)
    val inputWidth = input.size(dimw)
    val nInputPlane = input.size(dimh - 1)
    var outputHeight =
      if (ceilMode) {
        math.ceil((inputHeight - kH + 2 * padH).toFloat / dH).toInt + 1
      } else {
        math.floor((inputHeight - kH + 2 * padH).toFloat / dH).toInt + 1
      }
    var outputWidth =
      if (ceilMode) {
        math.ceil((inputWidth - kW + 2 * padW).toFloat / dW).toInt + 1
      } else {
        math.floor((inputWidth - kW + 2 * padW).toFloat / dW).toInt + 1
      }
    if (padW != 0 || padH != 0) {
      // ensure that the last pooling starts inside the image
      // needed to avoid problems in ceil mode
      if ((outputHeight - 1) * dH >= inputHeight + padH) {
        outputHeight -= 1
      }
      if ((outputWidth - 1) * dW >= inputWidth + padW) {
        outputWidth -= 1
      }
    }

    gradInput.resizeAs(input).zero()
    if (input.dim() == 3) {
      if (classTag[T] == classTag[Double]) {
        updateGradInputFrameDouble(gradInput.asInstanceOf[Tensor[Double]],
          gradOutput.asInstanceOf[Tensor[Double]],
          nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kW, kH, dW, dH)
      } else {
        updateGradInputFrameFloat(gradInput.asInstanceOf[Tensor[Float]],
          gradOutput.asInstanceOf[Tensor[Float]],
          nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kW, kH, dW, dH)
      }
    }
    else {
      val nBatch = input.size(1)

      if (results == null || results.length != nBatch) {
        results = new Array[Future[Unit]](nBatch)
      }

      var i = 1
      while (i <= nBatch) {
        val _i = i
        results(_i - 1) = Engine.model.invoke(() => {
          if (classTag[T] == classTag[Double]) {
            updateGradInputFrameDouble(gradInput(_i).asInstanceOf[Tensor[Double]],
              gradOutput(_i).asInstanceOf[Tensor[Double]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH)
          } else {
            updateGradInputFrameFloat(gradInput(_i).asInstanceOf[Tensor[Float]],
              gradOutput(_i).asInstanceOf[Tensor[Float]],
              nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
              kW, kH, dW, dH)
          }
        })
        i += 1
      }
      Engine.model.sync(results)
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
    s"nn.SpatialAveragePooling($kW, $kH, $dW, $dH, $padW, $padH)"
  }
}

object SpatialAveragePooling {
  def apply[@specialized(Float, Double) T: ClassTag](
      kW: Int,
      kH: Int,
      dW: Int = 1,
      dH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      ceilMode: Boolean = false,
      countIncludePad: Boolean = true,
      divide: Boolean = true)(implicit ev: TensorNumeric[T]) : SpatialAveragePooling[T] = {
    new SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH, ceilMode, countIncludePad, divide)
  }
}
