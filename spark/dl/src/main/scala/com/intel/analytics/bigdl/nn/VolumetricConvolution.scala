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

import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import org.apache.spark.sql.catalyst.optimizer.OptimizeIn

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
 * @param wRegularizer: instance of [[Regularizer]]
 * (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 * applied to the bias.
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class VolumetricConvolution[T: ClassTag](
  val nInputPlane: Int, val nOutputPlane: Int,
  val kT: Int, val kW: Int, val kH: Int,
  val dT: Int = 1, val dW: Int = 1, val dH: Int = 1,
  val padT: Int = 0, val padW: Int = 0, val padH: Int = 0, withBias: Boolean = true,
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

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

  {
    val stdv = 1.0 / math.sqrt(kT * kW * kH * nInputPlane)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)

    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.OUT_IN_KT_KH_KW)
    Option(bias).foreach(biasInitMethod.init(_, VariableFormat.ONE_D))
    zeroGradParameters()
  }

  override def clearState(): this.type = {
    super.clearState()
    fInput.set()
    fGradInput.set()
    if (withBias) onesBias.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (withBias) {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    } else {
      (Array(this.weight), Array(this.gradWeight))
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"Convolution3D requires 5D input, but got input dim ${input.length}")
    require(input(1) == nInputPlane, s"input.size(1) should be equal to nInputPlane. " +
      s"But In ${this.getName()} : input.size(1) is: ${ input(1) } ," +
      s" nInputPlane is: ${ nInputPlane }")
    val inputWidth = input(4)
    val inputHeight = input(3)
    val inputDepth = input(2)
    val sizes = if (padW == -1 && padH == -1 && padT == -1) {
      Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, inputDepth, dT, kT)
    } else {
      Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, padH, padW, ceilMode = false, inputdepth = inputDepth,
        dt = dT, kt = kT, padt = padT)
    }
    val outputDepth = sizes(6)
    val outputHeight = sizes(7)
    val outputWidth = sizes(8)
    require(outputWidth >= 1 && outputDepth >= 1 && outputHeight >= 1,
      s"Given input size: (${ input.mkString("x") })." +
        s" Calculated output size:" +
        s" (${ nOutputPlane }x${ outputDepth }x${ outputHeight }x${ outputWidth })." +
        s" Output size is too small")
    Shape(input(0), nOutputPlane, outputDepth, outputHeight, outputWidth)
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

    require(weight.dim() == 2 || weight.dim() == 5,
      s"weight tensor should be 2D or 5D - got ${ weight.dim() }")


    if (input.dim() == 4) {
      require(input.size(1) == nInputPlane, s"input.size(1) should be equal to nInputPlane. " +
        s"But In ${this.getName()} : input.size(1) is: ${ input.size(1) } ," +
        s" nInputPlane is: ${ nInputPlane }")
    }

    VolumetricConvolution.conv3d(input, output, weightMM, bias, onesBias, fInput,
      nInputPlane, nOutputPlane, withBias, kT, kW, kH, dT, dW, dH, padT, padW, padH)
    output
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

    VolumetricConvolution.conv3DBackpropInput(input, gradInput, gradOutput, weightMM,
      fGradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(gradOutput.isContiguous(), "gradOutput should be contiguous")
    if (gradWeightMM == null || gradWeightMM.storage().isEmpty) {
      gradWeightMM = gradWeight.view(nOutputPlane, nInputPlane * kT * kH * kW)
    }

    VolumetricConvolution.conv3DBackpropFilter(input, gradOutput, gradWeightMM, gradBias,
      fInput, scaleB, scaleW, withBias)

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (withBias && null != bRegularizer) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
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
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]): VolumetricConvolution[T] = {
    new VolumetricConvolution[T](nInputPlane, nOutputPlane, kT, kW, kH,
      dT, dW, dH, padT, padW, padH, withBias, wRegularizer, bRegularizer)
  }

  private[bigdl] def conv3d[T](input: Tensor[T],
                               output: Tensor[T],
                               weightMM: Tensor[T],
                               bias: Tensor[T],
                               onesBias: Tensor[T],
                               fInput: Tensor[T],
                               nInputPlane: Int,
                               nOutputPlane: Int,
                               withBias: Boolean,
                               kT: Int, kW: Int, kH: Int,
                               dT: Int, dW: Int, dH: Int,
                               padT: Int, padW: Int, padH: Int
                              )(implicit ev: TensorNumeric[T]): Unit = {
    val dimDepth = if (input.dim() == 4) 2 else 3
    val dimWidth = if (input.dim() == 4) 4 else 5
    val dimHeight = if (input.dim() == 4) 3 else 4

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)
    val inputDepth = input.size(dimDepth)

    val sizes = if (padW == -1 && padH == -1 && padT == -1) {
      Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, inputDepth, dT, kT)
    } else {
      Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, padH, padW, ceilMode = false, inputdepth = inputDepth,
        dt = dT, kt = kT, padt = padT)
    }
    val padFront = sizes(0)
    val padBack = sizes(1)
    val padLeft = sizes(4)
    val padRight = sizes(5)
    val padTop = sizes(2)
    val padBottom = sizes(3)
    val outputDepth = sizes(6)
    val outputHeight = sizes(7)
    val outputWidth = sizes(8)

    require(outputWidth >= 1 && outputDepth >= 1 && outputHeight >= 1,
      s"Given input size: (${ input.size().mkString("x") })." +
        s" Calculated output size:" +
        s" (${ nOutputPlane }x${ outputDepth }x${ outputHeight }x${ outputWidth })." +
        s" Output size is too small")

    if (withBias && (onesBias.dim() != 1 || onesBias.size(1) !=
      outputHeight * outputWidth * outputDepth)) {
      onesBias.resize(Array(outputHeight * outputWidth * outputDepth)).fill(ev.one)
    }

    if (input.dim() == 4) {
      fInput.resize(kT * kW * kH * nInputPlane, outputDepth * outputHeight * outputWidth)
      output.resize(nOutputPlane, outputDepth, outputHeight, outputWidth)
      updateOutputFrame(input, output, weightMM, bias, fInput, kT, kW, kH, dT, dW, dH,
        padFront, padLeft, padTop, padBack, padRight, padBottom, nInputPlane,
        inputDepth, inputWidth, inputHeight,
        nOutputPlane, outputDepth, outputWidth, outputHeight, withBias, onesBias)
    } else {
      fInput.resize(input.size(1), kT * kW * kH * nInputPlane,
        outputDepth * outputHeight * outputWidth)
      output.resize(input.size(1), nOutputPlane, outputDepth, outputHeight, outputWidth)

      var t = 1
      while (t <= input.size(1)) {
        val inputT = input.select(1, t)
        val outputT = output.select(1, t)
        val fInputT = fInput.select(1, t)
        updateOutputFrame(inputT, outputT, weightMM, bias, fInputT,
          kT, kW, kH,
          dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          nInputPlane, inputDepth, inputWidth, inputHeight,
          nOutputPlane, outputDepth, outputWidth, outputHeight, withBias, onesBias)
        t += 1
      }
    }
  }

  private def updateOutputFrame[T](
    input: Tensor[T], output: Tensor[T], weight: Tensor[T],
    bias: Tensor[T], fInput: Tensor[T], kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
    padFront: Int, padLeft: Int, padTop: Int, padBack: Int, padRight: Int, padBottom: Int,
    nInputPlane: Int, inputDepth: Int, inputWidth: Int, inputHeight: Int,
    nOutputPlane: Int, outputDepth: Int, outputWidth: Int, outputHeight: Int,
    withBias: Boolean, onesBias: Tensor[T])
                                  (implicit ev: TensorNumeric[T]): Unit = {
    val output2d = output.view(nOutputPlane, outputDepth * outputHeight * outputWidth)

    ev.getType() match {
      case DoubleType =>
        NNPrimitive.unfoldedCopyVolDouble(fInput.asInstanceOf[Tensor[Double]],
          input.asInstanceOf[Tensor[Double]], kT, kW, kH, dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          nInputPlane,
          inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight)
      case FloatType =>
        NNPrimitive.unfoldedCopyVolFloat(fInput.asInstanceOf[Tensor[Float]],
          input.asInstanceOf[Tensor[Float]], kT, kW, kH, dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          nInputPlane,
          inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight)
      case t => throw new NotImplementedError(s"$t is not supported")
    }

    output2d.addmm(ev.zero, output2d, ev.one, weight, fInput)
    if (withBias) {
      output2d.addr(ev.one, bias, onesBias)
    }
  }

  private[bigdl] def conv3DBackpropInput[T](inputSize: Array[Int],
                                            gradInput: Tensor[T],
                                            gradOutput: Tensor[T],
                                            weightMM: Tensor[T],
                                            fGradInput: Tensor[T],
                                            kT: Int, kW: Int, kH: Int,
                                            dT: Int, dW: Int, dH: Int,
                                            padT: Int, padW: Int, padH: Int
                                           )(implicit ev: TensorNumeric[T]): Unit = {
    val dimChannel = if (inputSize.length == 4) 1 else 2
    val dimDepth = if (inputSize.length == 4) 2 else 3
    val dimWidth = if (inputSize.length == 4) 4 else 5
    val dimHeight = if (inputSize.length == 4) 3 else 4

    val nInputPlane = inputSize(dimChannel - 1)
    val inputWidth = inputSize(dimWidth - 1)
    val inputHeight = inputSize(dimHeight - 1)
    val inputDepth = inputSize(dimDepth - 1)


    val outputDepth = gradOutput.size(dimDepth)
    val outputHeight = gradOutput.size(dimHeight)
    val outputWidth = gradOutput.size(dimWidth)

    val sizes = if (padW == -1 && padH == -1 && padT == -1) {
      Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, inputDepth, dT, kT)
    } else {
      Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, padH, padW, ceilMode = false, inputdepth = inputDepth,
        dt = dT, kt = kT, padt = padT)
    }
    val padFront = sizes(0)
    val padBack = sizes(1)
    val padLeft = sizes(4)
    val padRight = sizes(5)
    val padTop = sizes(2)
    val padBottom = sizes(3)

    gradInput.resize(inputSize)

    if (inputSize.length == 4) {
      fGradInput.resize(kT * kW * kH * nInputPlane, outputDepth * outputHeight * outputWidth)
      require(gradOutput.isContiguous(), "gradOutput should be contiguous")
      updateGradInputFrame(gradInput, gradOutput, weightMM.transpose(1, 2), fGradInput,
        kT, kW, kH,
        dT, dW, dH,
        padFront, padLeft, padTop, padBack, padRight, padBottom)
    } else {
      fGradInput.resize(inputSize(0), kT * kW * kH * nInputPlane,
        outputDepth * outputHeight * outputWidth)
      // batch mode
      var t = 1
      while (t <= inputSize(0)) {
        val gradInputT = gradInput.select(1, t)
        val gradOutputT = gradOutput.select(1, t)
        val fGradInputT = fGradInput.select(1, t)
        require(gradOutputT.isContiguous(), "each batch of gradOutput should be contiguous")
        updateGradInputFrame(gradInputT, gradOutputT, weightMM.transpose(1, 2), fGradInputT,
          kT, kW, kH,
          dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom)
        t += 1
      }
    }
  }

  private[bigdl] def conv3DBackpropInput[T](input: Tensor[T],
                                         gradInput: Tensor[T],
                                         gradOutput: Tensor[T],
                                         weightMM: Tensor[T],
                                         fGradInput: Tensor[T],
                                         kT: Int, kW: Int, kH: Int,
                                         dT: Int, dW: Int, dH: Int,
                                         padT: Int, padW: Int, padH: Int
                                        )(implicit ev: TensorNumeric[T]): Unit = {
    conv3DBackpropInput(input.size(), gradInput, gradOutput, weightMM, fGradInput,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)
  }

  private def updateGradInputFrame[T](
    gradInput: Tensor[T], gradOutput: Tensor[T], weight: Tensor[T],
    fGradInput: Tensor[T], kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
    padFront: Int, padLeft: Int, padTop: Int, padBack: Int, padRight: Int, padBottom: Int)
                                     (implicit ev: TensorNumeric[T]):
  Unit = {
    val gradOutput2d = gradOutput.view(gradOutput.size(1),
      gradOutput.size(2) * gradOutput.size(3) * gradOutput.size(4))
    fGradInput.addmm(ev.zero, fGradInput,
      ev.one, weight, gradOutput2d)

    gradInput.zero()
    ev.getType() match {
      case DoubleType =>
        NNPrimitive.unfoldedAccVolDouble(fGradInput.asInstanceOf[Tensor[Double]],
          gradInput.asInstanceOf[Tensor[Double]], kT, kW, kH, dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          gradInput.size(1), gradInput.size(2), gradInput.size(4), gradInput.size(3),
          gradOutput.size(2), gradOutput.size(4), gradOutput.size(3))
      case FloatType =>
        NNPrimitive.unfoldedAccVolFloat(fGradInput.asInstanceOf[Tensor[Float]],
          gradInput.asInstanceOf[Tensor[Float]], kT, kW, kH, dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          gradInput.size(1), gradInput.size(2), gradInput.size(4), gradInput.size(3),
          gradOutput.size(2), gradOutput.size(4), gradOutput.size(3))
      case t => throw new NotImplementedError(s"$t is not supported")
    }

  }

  private[bigdl] def populateFInput[T](
                               input: Tensor[T],
                               fInput: Tensor[T],
                               nInputPlane: Int,
                               nOutputPlane: Int,
                               kT: Int, kW: Int, kH: Int,
                               dT: Int, dW: Int, dH: Int,
                               padT: Int, padW: Int, padH: Int
                              )(implicit ev: TensorNumeric[T]): Unit = {
    val dimDepth = if (input.dim() == 4) 2 else 3
    val dimWidth = if (input.dim() == 4) 4 else 5
    val dimHeight = if (input.dim() == 4) 3 else 4

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)
    val inputDepth = input.size(dimDepth)

    val sizes = if (padW == -1 && padH == -1 && padT == -1) {
      Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, inputDepth, dT, kT)
    } else {
      Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH,
        dW, kH, kW, padH, padW, ceilMode = false, inputdepth = inputDepth,
        dt = dT, kt = kT, padt = padT)
    }
    val padFront = sizes(0)
    val padBack = sizes(1)
    val padLeft = sizes(4)
    val padRight = sizes(5)
    val padTop = sizes(2)
    val padBottom = sizes(3)
    val outputDepth = sizes(6)
    val outputHeight = sizes(7)
    val outputWidth = sizes(8)

    require(outputWidth >= 1 && outputDepth >= 1 && outputHeight >= 1,
      s"Given input size: (${ input.size().mkString("x") })." +
        s" Calculated output size:" +
        s" (${ nOutputPlane }x${ outputDepth }x${ outputHeight }x${ outputWidth })." +
        s" Output size is too small")


    if (input.dim() == 4) {
      fInput.resize(kT * kW * kH * nInputPlane, outputDepth * outputHeight * outputWidth)
      im2colWrapper(input, fInput, kT, kW, kH, dT, dW, dH,
        padFront, padLeft, padTop, padBack, padRight, padBottom, nInputPlane,
        inputDepth, inputWidth, inputHeight,
        nOutputPlane, outputDepth, outputWidth, outputHeight)
    } else {
      fInput.resize(input.size(1), kT * kW * kH * nInputPlane,
        outputDepth * outputHeight * outputWidth)

      var t = 1
      while (t <= input.size(1)) {
        val inputT = input.select(1, t)
        val fInputT = fInput.select(1, t)
        im2colWrapper(inputT, fInputT,
          kT, kW, kH,
          dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          nInputPlane, inputDepth, inputWidth, inputHeight,
          nOutputPlane, outputDepth, outputWidth, outputHeight)
        t += 1
      }
    }
  }

  private def im2colWrapper[T](
      input: Tensor[T],
      fInput: Tensor[T], kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
      padFront: Int, padLeft: Int, padTop: Int, padBack: Int, padRight: Int, padBottom: Int,
      nInputPlane: Int, inputDepth: Int, inputWidth: Int, inputHeight: Int,
      nOutputPlane: Int, outputDepth: Int, outputWidth: Int, outputHeight: Int)
                              (implicit ev: TensorNumeric[T]): Unit = {
    ev.getType() match {
      case DoubleType =>
        NNPrimitive.unfoldedCopyVolDouble(fInput.asInstanceOf[Tensor[Double]],
          input.asInstanceOf[Tensor[Double]], kT, kW, kH, dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          nInputPlane,
          inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight)
      case FloatType =>
        NNPrimitive.unfoldedCopyVolFloat(fInput.asInstanceOf[Tensor[Float]],
          input.asInstanceOf[Tensor[Float]], kT, kW, kH, dT, dW, dH,
          padFront, padLeft, padTop, padBack, padRight, padBottom,
          nInputPlane,
          inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }

  private[bigdl] def conv3DBackpropFilter[T](input: Tensor[T],
                              gradOutput: Tensor[T],
                              gradWeightMM: Tensor[T],
                              gradBias: Tensor[T],
                              fInput: Tensor[T],
                              scaleW: Double, scaleB: Double,
                              withBias: Boolean)
                             (implicit ev: TensorNumeric[T]): Unit = {

    if (input.dim() == 4) {
      accGradParametersFrame(gradOutput, gradWeightMM, gradBias, fInput,
        ev.fromType[Double](scaleW), ev.fromType[Double](scaleB), withBias)
    } else {
      // batch mode
      var t = 1
      while (t <= input.size(1)) {
        val gradOutputT = gradOutput.select(1, t)
        val fInputT = fInput.select(1, t)
        accGradParametersFrame(gradOutputT, gradWeightMM, gradBias, fInputT,
          ev.fromType[Double](scaleW), ev.fromType[Double](scaleB), withBias)
        t += 1
      }
    }
  }

  private def accGradParametersFrame[T](
       gradOutput: Tensor[T], gradWeight: Tensor[T], gradBias: Tensor[T],
       fInput: Tensor[T], scaleW: T, scaleB: T, withBias: Boolean)
       (implicit ev: TensorNumeric[T]): Unit = {
    val gradOutput2d = gradOutput.view(gradOutput.size(1), gradOutput.size(2) *
      gradOutput.size(3) * gradOutput.size(4))
    val fInputT = fInput.transpose(1, 2)
    if (scaleW != 0) {
      gradWeight.addmm(ev.one, gradWeight, scaleW, gradOutput2d, fInputT)
    }
    if (withBias && scaleB != 0) {
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
          ev.times(scaleB, sum)))
        i += 1
      }
    }
  }

}
