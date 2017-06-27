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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.Future
import scala.reflect.ClassTag

@SerialVersionUID(4479683852714800631L)
class SpatialShareConvolution[T: ClassTag](
  nInputPlane: Int, // The number of expected input planes in the image given into forward()
  nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  kernelW: Int, // The kernel width of the convolution
  kernelH: Int, // The kernel height of the convolution
  strideW: Int = 1, // The step of the convolution in the width dimension.
  strideH: Int = 1, // The step of the convolution in the height dimension
  padW: Int = 0, // The additional zeros added per width to the input planes.
  padH: Int = 0, // The additional zeros added per height to the input planes.
  nGroup: Int = 1, // Kernel group number
  propagateBack: Boolean = true // propagate gradient back
  )
  (implicit ev: TensorNumeric[T]) extends SpatialConvolution[T](
  nInputPlane, nOutputPlane, kernelW, kernelH, strideW, strideH,
  padW, padH, nGroup, propagateBack) {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialShareConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    if (weightMM == null) {
      weightMM = weight.view(nGroup, nOutputPlane / nGroup,
        nInputPlane * kernelH * kernelW / nGroup)
    }
    val (outputWidth, outputHeight, inputWidth, inputHeight) = calcOutputWH(input)
    if (onesBias.dim() != 1 || onesBias.size(1) != outputHeight * outputWidth) {
      onesBias.resize(Array(outputHeight * outputWidth)).fill(ev.fromType(1.0))
    }

    require(outputWidth >= 1 && outputHeight >= 1, "output size is too small")
    if (input.dim() == 3) {
      require(input.size(1) == nInputPlane)
      require(input.isContiguous())
      output.resize(Array(nOutputPlane, outputHeight, outputWidth))
      fInput.resize(Array(nGroup, kernelW * kernelH * nInputPlane / nGroup,
        outputHeight * outputWidth))
      var g = 0
      while (g < nGroup) {
        updateOutputFrame(
          input.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          output.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1),
          bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          fInput.select(1, g + 1),
          kernelW, kernelH, strideW, strideH,
          padW, padH,
          nInputPlane / nGroup, inputWidth, inputHeight,
          nOutputPlane / nGroup, outputWidth, outputHeight)
        g += 1
      }
    } else {
      require(input.size(2) == nInputPlane)
      val batchSize = input.size(1)
      output.resize(Array(batchSize, nOutputPlane, outputHeight, outputWidth))

      val coresNum = Math.min(batchSize, Engine.model.getPoolSize)
      fInput.resize(Array(coresNum, nGroup, kernelW * kernelH * nInputPlane / nGroup,
        outputHeight * outputWidth))

      if (results == null || results.length != coresNum) {
        results = new Array[Future[Unit]](coresNum)
      }

      var i, j = 0
      val minJobNum: Int = batchSize / Engine.model.getPoolSize
      val remainJobNum: Int = batchSize - minJobNum * Engine.model.getPoolSize

      while (j < coresNum) {
        val _j = j
        results(j) = Engine.model.invoke(() => {
          var _i = 1
          val distJobNum: Int = minJobNum + (if (_j < remainJobNum) 1 else 0)
          val indexStart: Int = _j * minJobNum + (if (_j < remainJobNum) _j else remainJobNum)
          while (_i <= distJobNum) {
            val inputT = input.select(1, _i + indexStart).contiguous()
            val outputT = output.select(1, _i + indexStart)
            val fInputT = fInput.select(1, _j + 1)
            var g = 0
            while (g < nGroup) {
              updateOutputFrame(
                inputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
                outputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                weightMM.select(1, g + 1),
                bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                fInputT.select(1, g + 1),
                kernelW, kernelH, strideW, strideH,
                padW, padH,
                nInputPlane / nGroup, inputWidth, inputHeight,
                nOutputPlane / nGroup, outputWidth, outputHeight)
              g += 1
            }
            _i += 1
          }
        })
        j += 1
      }
      Engine.model.sync(results)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!propagateBack) {
      return gradInput
    }

    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    gradInput.resizeAs(input)

    if (input.nDimension() == 3) {
      require(gradOutput.isContiguous())
      val (outputWidth, outputHeight, _, _) = calcOutputWH(input)
      fGradInput.resize(Array(nGroup,
        kernelW * kernelH * nInputPlane / nGroup, outputHeight * outputWidth))
      var g = 0
      while (g < nGroup) {
        updateGradInputFrame(
          gradInput.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          gradOutput.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          weightMM.select(1, g + 1).transpose(1, 2),
          fGradInput.select(1, g + 1),
          kernelW, kernelH, strideW, strideH, padW, padH)
        g += 1
      }
    } else {
      val batchSize = input.size(1)
      val (outputWidth, outputHeight, _, _) = calcOutputWH(input)
      fGradInput.resize(Array(Engine.model.getPoolSize, nGroup,
        kernelW * kernelH * nInputPlane / nGroup, outputHeight * outputWidth))

      val coresNum = Math.min(batchSize, Engine.model.getPoolSize)
      if (results == null || results.length != coresNum) {
        results = new Array[Future[Unit]](coresNum)
      }

      var i, j = 0
      val minJobNum: Int = batchSize / Engine.model.getPoolSize
      val remainJobNum: Int = batchSize - minJobNum * Engine.model.getPoolSize
      while (j < coresNum) {
        val _j = j
        results(j) = Engine.model.invoke(() => {
          var _i = 1
          val distJobNum: Int = minJobNum + (if (_j < remainJobNum) 1 else 0)
          val indexStart: Int = _j * minJobNum + (if (_j < remainJobNum) _j else remainJobNum)
          while (_i <= distJobNum) {
            val gradInputT = gradInput.select(1, _i + indexStart)
            val gradOutputT = gradOutput.select(1, _i + indexStart).contiguous()
            val fgradInputT = fGradInput.select(1, _j + 1)
            var g = 0
            while (g < nGroup) {
              updateGradInputFrame(
                gradInputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
                gradOutputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                weightMM.select(1, g + 1).transpose(1, 2),
                fgradInputT.select(1, g + 1),
                kernelW, kernelH, strideW, strideH, padW, padH)
              g += 1
            }
            _i += 1
          }
        })
        j += 1
      }
      Engine.model.sync(results)
    }
    return gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input")
    require(gradOutput.isContiguous())

    if (input.nDimension() == 3) {
      if (gradWeightMM == null) {
        gradWeightMM = gradWeight.view(nGroup, nOutputPlane / nGroup,
          nInputPlane * kernelH * kernelW / nGroup)
      }
      val (outputWidth, outputHeight, inputWidth, inputHeight) = calcOutputWH(input)
      fInput.resize(Array(nGroup,
        kernelW * kernelH * nInputPlane / nGroup, outputHeight * outputWidth))
      var g = 0
      while (g < nGroup) {
        write2fInput(
          input.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
          fInput.select(1, g + 1),
          kernelW, kernelH, strideW, strideH,
          padW, padH,
          nInputPlane / nGroup, inputWidth, inputHeight,
          nOutputPlane / nGroup, outputWidth, outputHeight)
        accGradParametersFrame(
          gradOutput.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          gradWeightMM.select(1, g + 1),
          gradBias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
          fInput.select(1, g + 1),
          ev.fromType[Double](scale))
        g += 1
      }
    } else {
      val batchSize = input.size(1)
      if (gradWeightMMInBatch == null) {
        gradWeightMMInBatch = Tensor[T]().resize(Array(batchSize, nGroup, nOutputPlane / nGroup,
          nInputPlane * kernelH * kernelW / nGroup))
      }
      if (gradientBiasMT.nElement() == 0) {
        gradientBiasMT.resize(Array(batchSize, nOutputPlane))
      }
      if (ones.dim() != 1 || ones.size(1) != gradOutput.size(3) * gradOutput.size(4)) {
        ones.resize(Array(gradOutput.size(3) * gradOutput.size(4))).fill(ev.fromType(1.0))
      }

      if (onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
        onesBatch.resize(Array(batchSize)).fill(ev.fromType(1.0))
      }
      val coresNum = Math.min(batchSize, Engine.model.getPoolSize)
      if (results == null || results.length != coresNum) {
        results = new Array[Future[Unit]](coresNum)
      }

      var i, j = 0
      val minJobNum: Int = batchSize / Engine.model.getPoolSize
      val remainJobNum: Int = batchSize - minJobNum * Engine.model.getPoolSize
      val (outputWidth, outputHeight, inputWidth, inputHeight) = calcOutputWH(input)
      fInput.resize(Array(Engine.model.getPoolSize, nGroup,
        kernelW * kernelH * nInputPlane / nGroup, outputHeight * outputWidth))
      while (j < coresNum) {
        val _j = j
        results(j) = Engine.model.invoke(() => {
          var _i = 1
          val distJobNum: Int = minJobNum + (if (_j < remainJobNum) 1 else 0)
          val indexStart: Int = _j * minJobNum + (if (_j < remainJobNum) _j else remainJobNum)
          while (_i <= distJobNum) {
            val gradOutputT = gradOutput.select(1, _i + indexStart)
            val inputT = input.select(1, _i + indexStart).contiguous()
            val fInputT = fInput.select(1, _j + 1)
            var g = 0
            while (g < nGroup) {
              write2fInput(
                inputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
                fInputT.select(1, g + 1),
                kernelW, kernelH, strideW, strideH,
                padW, padH,
                nInputPlane / nGroup, inputWidth, inputHeight,
                nOutputPlane / nGroup, outputWidth, outputHeight)
              calcGradParametersFrame(
                gradOutputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
                gradWeightMMInBatch.select(1, _i + indexStart).select(1, g + 1),
                gradientBiasMT.select(1, _i + indexStart).narrow(1, g * nOutputPlane / nGroup + 1,
                  nOutputPlane / nGroup),
                fInputT.select(1, g + 1),
                ev.fromType[Double](scale))
              g += 1
            }
            _i += 1
          }
        })
        j += 1
      }
      Engine.model.sync(results)

      val gradView = gradWeightMMInBatch.view(batchSize,
        nOutputPlane * nInputPlane * kernelH * kernelW / nGroup).t
      val grad = gradWeight.view(nOutputPlane * nInputPlane * kernelH * kernelW / nGroup)
      grad.addmv(ev.fromType(1.0), ev.fromType(1.0), gradView, onesBatch)
      gradBias.addmv(ev.fromType(1.0), ev.fromType(1.0), gradientBiasMT.t, onesBatch)
    }
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialShareConvolution[T]]) {
      return false
    }

    val other = obj.asInstanceOf[SpatialShareConvolution[T]]
    this.eq(other)
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

  override def clearState(): this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"${getPrintName}($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH)"
  }

  @inline
  private def calcOutputWH(input: Tensor[T]): (Int, Int, Int, Int) = {
    val dimWidth = if (input.dim() == 3) 3 else 4
    val dimHeight = if (input.dim() == 3) 2 else 3

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1
    val outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1

    require(outputWidth >= 1 && outputHeight >= 1, "output size is too small")

    (outputWidth, outputHeight, inputWidth, inputHeight)
  }

  @inline
  private def write2fInput(
    input: Tensor[T], fInput: Tensor[T],
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int,
    nOutputPlane: Int, outputWidth: Int, outputHeight: Int)(
    implicit ev: TensorNumeric[T]): Unit = {

    if (!_1x1) {
      ev.getType() match {
        case DoubleType =>
          val before = System.nanoTime()
          NNPrimitive.im2colDouble(fInput.asInstanceOf[Tensor[Double]],
            input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, nInputPlane,
            inputWidth, inputHeight, outputWidth, outputHeight)
          im2colTime += System.nanoTime() - before
        case FloatType =>
          val before = System.nanoTime()
          NNPrimitive.im2colFloat(fInput.asInstanceOf[Tensor[Float]],
            input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, nInputPlane,
            inputWidth, inputHeight, outputWidth, outputHeight)
          im2colTime += System.nanoTime() - before
        case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
    }
  }
}

object SpatialShareConvolution {
  def apply[@specialized(Float, Double) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true)
    (implicit ev: TensorNumeric[T]): SpatialShareConvolution[T] = {
    new SpatialShareConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup,
      propagateBack)
  }
}
