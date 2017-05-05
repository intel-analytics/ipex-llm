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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Applies a 3D convolution over an input image composed of several input planes. The input tensor
 * in forward(input) is expected to be a 4D tensor (nInputPlane x time x height x width).
 * @param nInputPlane The number of expected input planes in the image given into forward()
 * @param nOutputPlane The number of output planes the convolution layer will produce.
 * @param kT The kernel size of the convolution in time
 * @param kW The kernel width of the convolution
 * @param kH The kernel height of the convolution
 * @param dT The step of the convolution in the time dimension. Default is 1
 * @param dW The step of the convolution in the width dimension. Default is 1
 * @param dH The step of the convolution in the height dimension. Default is 1
 * @param padT Additional zeros added to the input plane data on both sides of time axis.
 * Default is 0. (kT-1)/2 is often used here.
 * @param padW The additional zeros added per width to the input planes.
 * @param padH The additional zeros added per height to the input planes.
 * @param withBias whether with bias
 * @param initMethod Init method, Default, Xavier, Bilinear.
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class VolumetricConvolution[T: ClassTag](
  val nInputPlane: Int, val nOutputPlane: Int,
  val kT: Int, val kW: Int, val kH: Int,
  val dT: Int = 1, val dW: Int = 1, val dH: Int = 1,
  val padT: Int = 0, val padW: Int = 0, val padH: Int = 0, withBias: Boolean = true,
  private var initMethod: InitializationMethod = Default
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(kT > 0 && kW > 0 && kH > 0, "kernel size should be greater than zero," +
    s" but got kT: $kT kH: $kH kW: $kW")
  require(dT > 0 && dW > 0 && dH > 0, "stride should be greater than zero," +
    s" but got dT: $dT dH: $dH dW: $dW")

  val weight: Tensor[T] = Tensor[T](nOutputPlane, nInputPlane, kT, kH, kW)
  val bias: Tensor[T] = if (withBias) Tensor[T](nOutputPlane) else null

  val gradWeight: Tensor[T] = Tensor[T](nOutputPlane, nInputPlane, kT, kH, kW)
  val gradBias: Tensor[T] = if (withBias) Tensor[T](nOutputPlane) else null

  val fInput = Tensor[T]()
  val fGradInput = Tensor[T]()

  private val onesBias = if (withBias) Tensor[T]() else null
  protected var weightMM: Tensor[T] = null
  protected var gradWeightMM: Tensor[T] = null

  reset()

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(kT * kW * kH * nInputPlane)
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        if (withBias) {
          bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        }
      case _ => throw new IllegalArgumentException()
    }
    zeroGradParameters()
  }

  override def clearState(): this.type = {
    super.clearState()
    fInput.set()
    fGradInput.set()
    if (withBias) onesBias.set()
    this
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    if (withBias) {
      bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    }
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    if (withBias) gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (withBias) {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    } else {
      (Array(this.weight), Array(this.gradWeight))
    }
  }

  override def getParametersTable(): Table = {
    if (withBias) {
      T(getName() -> T("weight" -> weight, "bias" -> bias,
        "gradWeight" -> gradWeight, "gradBias" -> gradBias))
    } else {
      T(getName() -> T("weight" -> weight,
        "gradWeight" -> gradWeight))
    }
  }

  private def updateOutputFrame(input: Tensor[T], output: Tensor[T], weight: Tensor[T],
    bias: Tensor[T], fInput: Tensor[T], kT: Int, kW: Int, kH: Int, dT: Int, dW: Int,
    dH: Int, pT: Int, pW: Int, pH: Int, nInputPlane: Int, inputDepth: Int,
    inputWidth: Int, inputHeight: Int, nOutputPlane: Int, outputDepth: Int, outputWidth: Int,
    outputHeight: Int): Unit = {
    val output2d = output.view(nOutputPlane, outputDepth * outputHeight * outputWidth)

    ev.getType() match {
      case DoubleType =>
        NNPrimitive.unfoldedCopyVolDouble(fInput.asInstanceOf[Tensor[Double]],
          input.asInstanceOf[Tensor[Double]], kT, kW, kH, dT, dW, dH, pT, pW, pH, nInputPlane,
          inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight)
      case FloatType =>
        NNPrimitive.unfoldedCopyVolFloat(fInput.asInstanceOf[Tensor[Float]],
          input.asInstanceOf[Tensor[Float]], kT, kW, kH, dT, dW, dH, pT, pW, pH, nInputPlane,
          inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight)
    }

    output2d.addmm(ev.zero, output2d, ev.one, weight, fInput)
    if (withBias) {
      output2d.addr(ev.one, bias, onesBias)
    }
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   * @param input
   * @return
   */
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous(), "input should be contiguous")
    require(input.dim() == 4 || input.dim() == 5,
      s"4D or 5D (batch mode) tensor expected for input, but got: ${ input.dim() }d")

    if (weightMM == null || weightMM.storage().isEmpty) {
      weightMM = weight.view(nOutputPlane, nInputPlane * kT * kH * kW)
    }

    val dimDepth = if (input.dim() == 4) 2 else 3
    val dimWidth = if (input.dim() == 4) 4 else 5
    val dimHeight = if (input.dim() == 4) 3 else 4

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)
    val inputDepth = input.size(dimDepth)

    val outputDepth = (inputDepth + 2 * padT - kT) / dT + 1
    val outputHeight = (inputHeight + 2 * padH - kH) / dH + 1
    val outputWidth = (inputWidth + 2 * padW - kW) / dW + 1

    require(outputWidth >= 1 && outputDepth >= 1 && outputHeight >= 1,
      s"Given input size: (${ input.size().mkString("x") })." +
        s" Calculated output size:" +
        s" (${ nOutputPlane }x${ outputDepth }x${ outputHeight }x${ outputWidth })." +
        s" Output size is too small")

    require(weight.dim() == 2 || weight.dim() == 5,
      s"weight tensor should be 2D or 5D - got ${ weight.dim() }")

    if (withBias && (onesBias.dim() != 1 || onesBias.size(1) !=
      outputHeight * outputWidth * outputDepth)) {
      onesBias.resize(Array(outputHeight * outputWidth * outputDepth)).fill(ev.one)
    }

    if (input.dim() == 4) {
      require(input.size(1) == nInputPlane, "input.size(1) should be equal to nInputPlane")
      fInput.resize(kT * kW * kH * nInputPlane, outputDepth * outputHeight * outputWidth)
      output.resize(nOutputPlane, outputDepth, outputHeight, outputWidth)
      updateOutputFrame(input, output, weightMM, bias, fInput, kT, kW, kH, dT, dW, dH,
        padT, padW, padH, nInputPlane, inputDepth, inputWidth, inputHeight,
        nOutputPlane, outputDepth, outputWidth, outputHeight)
    } else {
      fInput.resize(input.size(1), kT * kW * kH * nInputPlane,
        outputDepth * outputHeight * outputWidth)
      output.resize(input.size(1), nOutputPlane, outputDepth, outputHeight, outputWidth)

      var t = 1
      while (t < input.size(1)) {
        val inputT = input.select(1, t)
        val outputT = output.select(1, t)
        val fInputT = fInput.select(1, t)
        updateOutputFrame(inputT, outputT, weightMM, bias, fInputT,
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH,
          nInputPlane, inputDepth, inputWidth, inputHeight,
          nOutputPlane, outputDepth, outputWidth, outputHeight)
        t += 1
      }
    }
    output
  }

  private def updateGradInputFrame(gradInput: Tensor[T], gradOutput: Tensor[T], weight: Tensor[T],
    fGradInput: Tensor[T], kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
    pT: Int, pW: Int, pH: Int): Unit = {
    val gradOutput2d = gradOutput.view(gradOutput.size(1),
      gradOutput.size(2) * gradOutput.size(3) * gradOutput.size(4))
    fGradInput.addmm(ev.zero, fGradInput,
      ev.one, weight, gradOutput2d)

    gradInput.zero()
    ev.getType() match {
      case DoubleType =>
        NNPrimitive.unfoldedAccVolDouble(fGradInput.asInstanceOf[Tensor[Double]],
          gradInput.asInstanceOf[Tensor[Double]], kT, kW, kH, dT, dW, dH, pT, pW, pH,
          gradInput.size(1), gradInput.size(2), gradInput.size(4), gradInput.size(3),
          gradOutput.size(2), gradOutput.size(4), gradOutput.size(3))
      case FloatType =>
        NNPrimitive.unfoldedAccVolFloat(fGradInput.asInstanceOf[Tensor[Float]],
          gradInput.asInstanceOf[Tensor[Float]], kT, kW, kH, dT, dW, dH, pT, pW, pH,
          gradInput.size(1), gradInput.size(2), gradInput.size(4), gradInput.size(3),
          gradOutput.size(2), gradOutput.size(4), gradOutput.size(3))
    }

  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 4 || input.dim() == 5,
      s"4D or 5D (batch mode) tensor expected for input, but got: ${ input.dim() }d")
    require(input.isContiguous(), "input should be contiguous")
    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    gradInput.resizeAs(input)
    fGradInput.resizeAs(fInput).zero()
    if (input.dim() == 4) {
      updateGradInputFrame(gradInput, gradOutput, weightMM.transpose(1, 2), fGradInput,
        kT, kW, kH,
        dT, dW, dH,
        padT, padW, padH)
    } else {
      // batch mode
      var t = 1
      while (t < input.size(1)) {
        val gradInputT = gradInput.select(1, t)
        val gradOutputT = gradOutput.select(1, t)
        val fGradInputT = fGradInput.select(1, t)
        updateGradInputFrame(gradInputT, gradOutputT, weightMM.transpose(1, 2), fGradInputT,
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH)
        t += 1
      }
    }
    gradInput
  }

  def accGradParametersFrame(gradOutput: Tensor[T], gradWeight: Tensor[T], gradBias: Tensor[T],
    fInput: Tensor[T], scale: Double): Unit = {
    val gradOutput2d = gradOutput.view(gradOutput.size(1), gradOutput.size(2) *
      gradOutput.size(3) * gradOutput.size(4))
    val fInputT = fInput.transpose(1, 2)
    gradWeight.addmm(ev.one, gradWeight, ev.fromType(scale), gradOutput2d, fInputT)
    if (withBias) {
      var i = 0
      while (i < gradBias.size(1)) {
        var sum = ev.zero
        val data = gradOutput2d.storage().array()
        val offset = gradOutput2d.storageOffset() - 1 + i * gradOutput2d.stride(1)
        var k = 0
        while (k < gradOutput2d.size(2)) {
          sum = ev.plus(sum, data(k + offset))
          k += 1
        }
        gradBias.setValue(i + 1, ev.plus(gradBias.valueAt(i + 1),
          ev.times(ev.fromType(scale), sum)))
        i += 1
      }
    }
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {
    require(input.isContiguous(), "input should be contiguous")
    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    if (gradWeightMM == null || gradWeightMM.storage().isEmpty) {
      gradWeightMM = gradWeight.view(nOutputPlane, nInputPlane * kT * kH * kW)
    }
    if (input.dim() == 4) {
      accGradParametersFrame(gradOutput, gradWeightMM, gradBias, fInput, scale)
    } else {
      // batch mode
      var t = 1
      while (t < input.size(1)) {
        val gradOutputT = gradOutput.select(1, t)
        val fInputT = fInput.select(1, t)
        accGradParametersFrame(gradOutputT, gradWeightMM, gradBias, fInputT, scale)
        t += 1
      }
    }
  }

  override def toString: String = {
    s"nn.VolumetricConvolution($nInputPlane -> $nOutputPlane, $kT x $kW x" +
      s" $kH, $dT, $dW, $dH, $padT, $padW, $padH)"
  }
}

object VolumetricConvolution {
  def apply[@specialized(Float, Double) T: ClassTag](
    nInputPlane: Int, nOutputPlane: Int,
    kT: Int, kW: Int, kH: Int,
    dT: Int = 1, dW: Int = 1, dH: Int = 1,
    padT: Int = 0, padW: Int = 0, padH: Int = 0, withBias: Boolean = true,
    initMethod: InitializationMethod = Default
  )(implicit ev: TensorNumeric[T]): VolumetricConvolution[T] = {
    new VolumetricConvolution[T](nInputPlane, nOutputPlane, kT, kW, kH,
      dT, dW, dH, padT, padW, padH, withBias, initMethod)
  }
}
