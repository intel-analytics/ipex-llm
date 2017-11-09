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

import com.intel.analytics.bigdl.{Module, _}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Engine

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
        propagateBack: Boolean = true, // propagate gradient back
        wRegularizer: Regularizer[T] = null,
        bRegularizer: Regularizer[T] = null,
        initWeight: Tensor[T] = null,
        initBias: Tensor[T] = null,
        initGradWeight: Tensor[T] = null,
        initGradBias: Tensor[T] = null,
        withBias: Boolean = true
      )(implicit ev: TensorNumeric[T]) extends SpatialConvolution[T](
  nInputPlane, nOutputPlane, kernelW, kernelH, strideW, strideH,
  padW, padH, nGroup, propagateBack, wRegularizer, bRegularizer,
  initWeight, initBias, initGradWeight, initGradBias, withBias) {

  require(Engine.model.getPoolSize == 1, "Don't support single model multi thread.")
  require(padW >= 0 && padH >= 0, "SAME padding is not supported in SpatialShareConvolution," +
    " padW and padH should not be negative" +
    s"padW $padW padH $padH")

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialShareConvolution: " + ErrorInfo.constrainInputAs3DOrBatch +
    s"input dimension ${input.dim()}}")
    require(input.isContiguous())
    if (_1x1 || input.dim() == 3 || (input.dim() == 4 && input.size(1) == 1)) {
      super.updateOutput(input)
    } else {
      if (weightMM == null || weightMM.storage().isEmpty) {
        weightMM = weight.view(nGroup, nOutputPlane / nGroup,
          nInputPlane * kernelH * kernelW / nGroup)
      }

      val (outputWidth, outputHeight, inputWidth, inputHeight) = calcOutputWH(input)

      require(outputWidth >= 1 && outputHeight >= 1,
        s"output size is too small. outputWidth: $outputWidth, outputHeight: $outputHeight")

      if (withBias && onesBias.dim() != 1 || onesBias.size(1) != outputHeight * outputWidth) {
        onesBias.resize(outputHeight * outputWidth).fill(ev.fromType(1.0))
      }
      require(input.size(2) == nInputPlane)
      val batchSize = input.size(1)
      output.resize(batchSize, nOutputPlane, outputHeight, outputWidth)
      fInput.resize(nGroup, kernelW * kernelH * nInputPlane / nGroup,
        outputHeight * outputWidth)

      var i = 1
      while (i <= batchSize) {
        val inputT = input.select(1, i)
        require(inputT.isContiguous())
        val outputT = output.select(1, i)
        var g = 0
        while (g < nGroup) {
          val biasUse = if (withBias) {
            bias.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup)
          } else null
          updateOutputFrame(
            inputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
            outputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
            weightMM.select(1, g + 1),
            biasUse,
            fInput.select(1, g + 1),
            kernelW, kernelH, strideW, strideH,
            padW, padH, padW, padH,
            nInputPlane / nGroup, inputWidth, inputHeight,
            nOutputPlane / nGroup, outputWidth, outputHeight)
          g += 1
        }
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!propagateBack) {
      return gradInput
    }
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input" +
      s"input dimension ${input.nDimension()}")
    if (_1x1 || input.dim() == 3 || (input.dim() == 4 && input.size(1) == 1)) {
      super.updateGradInput(input, gradOutput)
    } else {
    gradInput.resizeAs(input)
      fGradInput.resizeAs(fInput)
      val batchSize = input.size(1)
      var i = 1
      while (i <= batchSize) {
          val gradInputT = gradInput.select(1, i)
          val gradOutputT = gradOutput.select(1, i)
          require(gradOutputT.isContiguous())
          var g = 0
          while (g < nGroup) {
            updateGradInputFrame(
              gradInputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
              gradOutputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              weightMM.select(1, g + 1).transpose(1, 2),
              fGradInput.select(1, g + 1),
              kernelW, kernelH, strideW, strideH, padW, padH, padW, padH)
            g += 1
          }
        i += 1
      }
    }

    return gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.nDimension() == 3 || input.nDimension() == 4, "Only support 3D or 4D input" +
      s"input Dimension ${input.nDimension()}")
    require(gradOutput.isContiguous())
    if (_1x1 || input.dim() == 3 || (input.dim() == 4 && input.size(1) == 1)) {
      super.accGradParameters(input, gradOutput)
    } else {
      val batchSize = input.size(1)
      if (gradWeightMMInBatch == null) {
        gradWeightMMInBatch = Tensor[T](batchSize, nGroup, nOutputPlane / nGroup,
          nInputPlane * kernelH * kernelW / nGroup)
      }
      if (withBias && gradientBiasMT.nElement() == 0) {
        gradientBiasMT.resize(batchSize, nOutputPlane)
      }
      if (ones.dim() != 1 || ones.size(1) != gradOutput.size(3) * gradOutput.size(4)) {
        ones.resize(gradOutput.size(3) * gradOutput.size(4)).fill(ev.fromType(1.0))
      }

      if (onesBatch.dim() != 1 || onesBatch.size(1) != batchSize) {
        onesBatch.resize(batchSize).fill(ev.fromType(1.0))
      }
      val (outputWidth, outputHeight, inputWidth, inputHeight) = calcOutputWH(input)
      var i = 1
      while (i <= batchSize) {
        val inputT = input.select(1, i)
          val gradOutputT = gradOutput.select(1, i)
          var g = 0
          while (g < nGroup) {
            write2fInput(
              inputT.narrow(1, g * nInputPlane / nGroup + 1, nInputPlane / nGroup),
              fInput.select(1, g + 1),
              kernelW, kernelH, strideW, strideH,
              padW, padH,
              nInputPlane / nGroup, inputWidth, inputHeight,
              nOutputPlane / nGroup, outputWidth, outputHeight)

            val gradientBiasMTUse = if (withBias) {
              gradientBiasMT.select(1, i).narrow(1, g * nOutputPlane / nGroup + 1,
                nOutputPlane / nGroup)
            } else null
            calcGradParametersFrame(
              gradOutputT.narrow(1, g * nOutputPlane / nGroup + 1, nOutputPlane / nGroup),
              gradWeightMMInBatch.select(1, i).select(1, g + 1),
              gradientBiasMTUse,
              fInput.select(1, g + 1),
              ev.fromType[Double](scaleW),
              ev.fromType[Double](scaleB)
            )
            g += 1
          }
        i += 1
      }

      val gradView = gradWeightMMInBatch.view(batchSize,
        nOutputPlane * nInputPlane * kernelH * kernelW / nGroup).t
      val grad = gradWeight.view(nOutputPlane * nInputPlane * kernelH * kernelW / nGroup)
      grad.addmv(ev.fromType(1.0), ev.fromType(1.0), gradView, onesBatch)
      if (withBias) {
        gradBias.addmv(ev.fromType(1.0), ev.fromType(1.0), gradientBiasMT.t, onesBatch)
      }
      if (null != wRegularizer) {
        wRegularizer.accRegularization(weight, gradWeight, scaleW)
      }
      if (withBias && null != bRegularizer) {
        bRegularizer.accRegularization(bias, gradBias, scaleB)
      }
    }
  }

  @inline
  private def calcOutputWH(input: Tensor[T]): (Int, Int, Int, Int) = {
    val dimWidth = if (input.dim() == 3) 3 else 4
    val dimHeight = if (input.dim() == 3) 2 else 3

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1
    val outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1

    require(outputWidth >= 1 && outputHeight >= 1, "output size is too small" +
      s"outputSize(${outputWidth},${outputHeight})")

    (outputWidth, outputHeight, inputWidth, inputHeight)
  }

  @inline
  private def write2fInput(
                            input: Tensor[T], fInput: Tensor[T],
                            kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
                            nInputPlane: Int, inputWidth: Int, inputHeight: Int,
                            nOutputPlane: Int, outputWidth: Int, outputHeight: Int)(
                            implicit ev: TensorNumeric[T]): Unit = {

    ev.getType() match {
      case DoubleType =>
        val before = System.nanoTime()
        NNPrimitive.im2colDouble(fInput.asInstanceOf[Tensor[Double]],
          input.asInstanceOf[Tensor[Double]], kW, kH, dW, dH, padW, padH, padW, padH,
          outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
      case FloatType =>
        val before = System.nanoTime()
        NNPrimitive.im2colFloat(fInput.asInstanceOf[Tensor[Float]],
          input.asInstanceOf[Tensor[Float]], kW, kH, dW, dH, padW, padH, padW, padH,
          outputWidth, outputHeight)
        im2colTime += System.nanoTime() - before
      case _ => throw new UnsupportedOperationException(s"Only Float/Double supported")
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
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    initGradWeight: Tensor[T] = null,
    initGradBias: Tensor[T] = null,
    withBias: Boolean = true)
    (implicit ev: TensorNumeric[T]): SpatialShareConvolution[T] = {
    new SpatialShareConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup, propagateBack, wRegularizer, bRegularizer,
      initWeight, initBias, initGradWeight, initGradBias, withBias)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
        conv: SpatialConvolution[T]
      )(implicit ev: TensorNumeric[T]): SpatialShareConvolution[T] = {
    val sConv = new SpatialShareConvolution[T](conv.nInputPlane, conv.nOutputPlane,
      conv.kernelW, conv.kernelH,
      conv.strideW, conv.strideH,
      conv.padW, conv.padH,
      conv.nGroup, conv.propagateBack,
      conv.wRegularizer, conv.bRegularizer, withBias = conv.withBias
    )
    sConv.weight.copy(conv.weight)
    sConv.gradWeight.copy(conv.gradWeight)
    if (conv.withBias) {
      sConv.gradBias.copy(conv.gradBias)
      sConv.bias.copy(conv.bias)
    }
    sConv.setScaleW(conv.getScaleW())
    sConv.setScaleB(conv.getScaleB())
    sConv.setName(conv.getName())
    sConv
  }

  /**
   * Replace all the SpatialConvolution in `model` with SpatialSharedConvolution,
   * and shared the fInput and fGradInput in all SpatialSharedConvolution.
   * @param model a Module
   * @return model sharedConvolution.
   */
  def shareConvolution[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val fInputCache = Tensor[T](1)
    val fGradInputCache = Tensor[T](1)
    shareConvolution(model, fInputCache, fGradInputCache)
    model
  }

  private def shareConvolution[T: ClassTag](
        model: Module[T],
        fInputCache: Tensor[T],
        fGradInputCache: Tensor[T])(implicit ev: TensorNumeric[T]): Unit = {
    model match {
      case container: Container[Activity, Activity, T] =>
        var i = 0
        while (i < container.modules.length) {
          val m = container.modules(i)
          if (m.isInstanceOf[SpatialConvolution[T]]) {
            val curModel = if (!m.isInstanceOf[SpatialShareConvolution[T]]) {
              SpatialShareConvolution(
                m.asInstanceOf[SpatialConvolution[T]])
            } else {
              m.asInstanceOf[SpatialShareConvolution[T]]
            }
            curModel.fInput.set(fInputCache)
            curModel.fGradInput.set(fGradInputCache)
            container.modules(i) = curModel
          } else {
            shareConvolution(m, fInputCache, fGradInputCache)
          }
          i += 1
        }
      case _ => Unit
    }
  }
}
