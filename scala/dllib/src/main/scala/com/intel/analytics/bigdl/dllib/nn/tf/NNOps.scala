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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.{Sigmoid, SpatialAveragePooling, SpatialBatchNormalization,
SpatialConvolution, SpatialCrossMapLRN, SpatialMaxPooling, SpatialSeparableConvolution, Tanh, Utils,
VolumetricConvolution, ELU => ELULayer, ReLU6 => ReLU6Layer, SoftPlus => SoftPlusLayer,
SoftSign => SoftSignLayer, ReLU => ReLULayer}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.ops.{ModuleToOperation, Operation}
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

private[bigdl] class Conv2D[T: ClassTag](
  strideH: Int,
  strideW: Int,
  padH: Int,
  padW: Int,
  format: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private var conv: SpatialConvolution[T] = _

  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)


    val channelDim = if (format == DataFormat.NHWC) 4 else 2
    val kHDim = if (format == DataFormat.NHWC) 1 else 3
    val kWDim = if (format == DataFormat.NHWC) 2 else 4

    if (conv == null) {
      conv = SpatialConvolution(
        nInputPlane = input.size(channelDim),
        nOutputPlane = filter.size(channelDim),
        kernelH = filter.size(kHDim),
        kernelW = filter.size(kWDim),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        withBias = false,
        format = format
      )
    }

    conv.setWeightsBias(Array(filter))
    output = conv.forward(input)
    output
  }
}

private[bigdl] object Conv2D {
  def apply[T: ClassTag](
    strideH: Int,
    strideW: Int,
    padH: Int,
    padW: Int,
    format: DataFormat = DataFormat.NHWC
  )(implicit ev: TensorNumeric[T]): Conv2D[T]
  = new Conv2D(strideH, strideW, padH, padW, format)
}

/**
 * Backward of SpatialConvolution
 */
private[bigdl] class Conv2DTranspose[T: ClassTag](
  strideW: Int,
  strideH: Int,
  padW: Int = -1,
  padH: Int = -1,
  format: DataFormat = DataFormat.NCHW
)(implicit ev: TensorNumeric[T])
  extends Operation[Activity, Tensor[T], T]{

  private var module: SpatialConvolution[T] = _
  private var dummyInput: Tensor[T] = _

  override def updateOutput(input: Activity): Tensor[T] = {
    require(input.isTable, "Invalid input activity type")
    val inputSizes = input.toTable.apply[Tensor[Int]](1).squeeze()
    val kernel = input.toTable.apply[Tensor[T]](2)
    val data = input.toTable.apply[Tensor[T]](3)

    require(data.nDimension() == 4, s"Need a 4D input but is ${data.nDimension()}")
    require(inputSizes.nDimension() == 1, s"Need a 1D size but is ${inputSizes.nDimension()}")

    val (nOutputPlane, nInputPlane) = if (format == DataFormat.NCHW) {
      (data.size(2), inputSizes.valueAt(2))
    } else {
      (data.size(4), inputSizes.valueAt(4))
    }

    val kHDim = if (format == DataFormat.NHWC) 1 else 3
    val kWDim = if (format == DataFormat.NHWC) 2 else 4


    if (module == null) {
      module = new SpatialConvolution[T](
        nInputPlane = nInputPlane,
        nOutputPlane = nOutputPlane,
        kernelW = kernel.size(kWDim),
        kernelH = kernel.size(kHDim),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        initWeight = kernel,
        format = format,
        withBias = false
      )

      dummyInput = Tensor[T](inputSizes.valueAt(1), inputSizes.valueAt(2), inputSizes.valueAt(3),
        inputSizes.valueAt(4))
    } else {
      val (nOutputPlanbe, nInputPlane) = if (format == DataFormat.NCHW) {
        (data.size(2), inputSizes.valueAt(2))
      } else {
        (data.size(4), inputSizes.valueAt(4))
      }

      require(module.nInputPlane == nInputPlane, "nInputPlane is not valid")
      require(module.nOutputPlane == nOutputPlane, "nOutputPlane is not valid")
      require(module.kernelH == kernel.size(kWDim), "kernelH is not valid")
      require(module.kernelW == kernel.size(kWDim), "kernelW is not valid")
      require(kernel.size(3) == nInputPlane, "kernel nInputPlane is not valid")
      require(kernel.size(4) == nOutputPlane, "kernel nOutputPlane is not valid")
      require(dummyInput.size(1) == inputSizes.valueAt(1), "size 1 is not correct")
      require(dummyInput.size(2) == inputSizes.valueAt(2), "size 1 is not correct")
      require(dummyInput.size(3) == inputSizes.valueAt(3), "size 1 is not correct")
      require(dummyInput.size(4) == inputSizes.valueAt(4), "size 1 is not correct")
    }

    module.forward(dummyInput)
    module.weight.set(kernel)
    module.updateGradInput(dummyInput, data)
    output = module.gradInput
    output
  }
}

private[bigdl] object Conv2DTranspose {
  def apply[T: ClassTag](
    strideW: Int,
    strideH: Int,
    padW: Int = -1,
    padH: Int = -1,
    format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): Conv2DTranspose[T] =
    new Conv2DTranspose(strideW, strideH, padW, padH, format)
}

private[bigdl] class Conv2DBackFilter[T: ClassTag](
  strideW: Int,
  strideH: Int,
  padW: Int = -1,
  padH: Int = -1,
  format: DataFormat = DataFormat.NCHW
)(implicit ev: TensorNumeric[T])
  extends Operation[Activity, Tensor[T], T]{

  private var module: SpatialConvolution[T] = _
  private var gradWeight: Tensor[T] = _
  private var dummyInput: Tensor[T] = _

  override def updateOutput(input: Activity): Tensor[T] = {
    require(input.isTable, "Invalid input activity type")
    val kernelSize = input.toTable.apply[Tensor[Int]](2).squeeze()
    val inputActivity = input.toTable.apply[Tensor[T]](1)
    val grads = input.toTable.apply[Tensor[T]](3)

    require(grads.nDimension() == 4, s"Need a 4D input but is ${grads.nDimension()}")
    require(kernelSize.nDimension() == 1, s"Need a 1D size but is ${kernelSize.nDimension()}")

    val (nOutputPlane, nInputPlane) = if (format == DataFormat.NCHW) {
      (grads.size(2), inputActivity.size(2))
    } else {
      (grads.size(4), inputActivity.size(4))
    }

    if (module == null) {
      gradWeight = Tensor[T]().resize(kernelSize.valueAt(1), kernelSize.valueAt(2),
        kernelSize.valueAt(3), kernelSize.valueAt(4))
      module = new SpatialConvolution[T](
        nInputPlane = nInputPlane,
        nOutputPlane = nOutputPlane,
        kernelW = kernelSize.valueAt(2),
        kernelH = kernelSize.valueAt(1),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        initGradWeight = gradWeight,
        format = format,
        withBias = false
      )
    } else {
      val (nOutputPlane, nInputPlane) = if (format == DataFormat.NCHW) {
        (grads.size(2), inputActivity.size(2))
      } else {
        (grads.size(4), inputActivity.size(4))
      }

      require(module.nInputPlane == nInputPlane, "nInputPlane is not valid")
      require(module.nOutputPlane == nOutputPlane, "nOutputPlane is not valid")
      require(module.kernelH == kernelSize.valueAt(1), s"kernelH is not valid")
      require(module.kernelW == kernelSize.valueAt(2), "kernelW is not valid")
      require(kernelSize.valueAt(3) == nInputPlane, "kernel nInputPlane is not valid")
      require(kernelSize.valueAt(4) == nOutputPlane, "kernel nOutputPlane is not valid")
    }

    module.forward(inputActivity)
    gradWeight.zero()
    module.accGradParameters(inputActivity, grads)
    output = module.gradWeight
    output
  }
}

private[bigdl] object Conv2DBackFilter {
  def apply[T: ClassTag](
    strideW: Int,
    strideH: Int,
    padW: Int = -1,
    padH: Int = -1,
    format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): Conv2DBackFilter[T] =
    new Conv2DBackFilter(strideW, strideH, padW, padH, format)
}

private[bigdl] class Conv3D[T: ClassTag](
  dT: Int, dH: Int, dW: Int,
  padT: Int, padH: Int, padW: Int,
  format: DataFormat)
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val fInput = Tensor[T]()


  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)

    val kT = filter.size(1)
    val kH = filter.size(2)
    val kW = filter.size(3)
    val nInputPlane = filter.size(4)
    val nOutputPlane = filter.size(5)

    val transInput = if (format == DataFormat.NHWC) {
      var buffer = input
      buffer = buffer.transpose(2, 5)
      buffer = buffer.transpose(3, 5)
      buffer = buffer.transpose(4, 5)
      buffer = buffer.contiguous()

      buffer
    } else {
      input
    }

    var transWeight = filter.transpose(1, 5)
    transWeight = transWeight.transpose(2, 4)
    transWeight = transWeight.transpose(3, 5)
    transWeight = transWeight.contiguous()
    val weightMM = transWeight.view(nOutputPlane, nInputPlane * kT * kH * kW)

    VolumetricConvolution.conv3d(transInput, output, weightMM, bias = null, onesBias = null, fInput,
      nInputPlane, nOutputPlane, withBias = false, kT, kW, kH, dT, dW, dH, padT, padW, padH)

    if (format == DataFormat.NHWC) {
      output = output.transpose(2, 5)
      output = output.transpose(2, 4)
      output = output.transpose(2, 3)
      output = output.contiguous()
    }
    output
  }

  override def clearState(): Conv3D.this.type = {
    super.clearState()
    fInput.set()
    this
  }
}

private[bigdl] object Conv3D {
  def apply[T: ClassTag](
    dT: Int,
    dH: Int,
    dW: Int,
    padT: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): Conv3D[T]
  = new Conv3D[T](dT, dH, dW, padT, padH, padW, format)
}

private[bigdl] class Conv3DBackpropFilter[T: ClassTag](
  dT: Int,
  dH: Int,
  dW: Int,
  padT: Int,
  padH: Int,
  padW: Int,
  format: DataFormat
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val fInput = Tensor[T]()


  protected def getParams(inputs: Table): (Int, Int, Int, Int, Int) = {
    val filter: Tensor[T] = inputs[Tensor[T]](2)

    val kT = filter.size(1)
    val kH = filter.size(2)
    val kW = filter.size(3)
    val nInputPlane = filter.size(4)
    val nOutputPlane = filter.size(5)

    (kT, kH, kW, nInputPlane, nOutputPlane)
  }
  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val outputBackprop: Tensor[T] = inputs[Tensor[T]](3)

    val (transInput, transOutBackprop) = if (format == DataFormat.NHWC) {
      // backpropInput only use input size, so we do not need it to be contiguous
      val in = input.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
      val out = outputBackprop.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
      (in, out)
    } else {
      (input, outputBackprop)
    }

    val (kT, kH, kW, nInputPlane, nOutputPlane) = getParams(inputs)

    val gradWeightMM = Tensor[T](nOutputPlane, nInputPlane * kT * kH * kW)

    VolumetricConvolution.populateFInput(transInput, fInput, nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)

    VolumetricConvolution.conv3DBackpropFilter(transInput, transOutBackprop, gradWeightMM,
      null, fInput, 1.0, 1.0, false)

    output = if (format == DataFormat.NHWC) {
      val gradWeight = gradWeightMM.view(nOutputPlane, nInputPlane, kT, kH, kW)
      gradWeight.transpose(1, 5).transpose(2, 4).transpose(1, 3).contiguous()
    } else {
      gradWeightMM.view(nOutputPlane, nInputPlane, kT, kH, kW)
    }

    output
  }

  override def clearState(): Conv3DBackpropFilter.this.type = {
    super.clearState()
    fInput.set()
    this
  }
}

private[bigdl] object Conv3DBackpropFilter {
  def apply[T: ClassTag](
    dT: Int,
    dH: Int,
    dW: Int,
    padT: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): Conv3DBackpropFilter[T]
  = new Conv3DBackpropFilter[T](dT, dH, dW, padT, padH, padW, format)
}

private[bigdl] class Conv3DBackpropFilterV2[T: ClassTag](
  dT: Int,
  dH: Int,
  dW: Int,
  padT: Int,
  padH: Int,
  padW: Int,
  format: DataFormat)
  (implicit ev: TensorNumeric[T])
  extends Conv3DBackpropFilter[T](dT, dH, dW, padT, padH, padW, format) {

  override protected def getParams(inputs: Table): (Int, Int, Int, Int, Int) = {
    val filterSize: Tensor[Int] = inputs[Tensor[Int]](2)

    val kT = filterSize.valueAt(1)
    val kH = filterSize.valueAt(2)
    val kW = filterSize.valueAt(3)
    val nInputPlane = filterSize.valueAt(4)
    val nOutputPlane = filterSize.valueAt(5)

    (kT, kH, kW, nInputPlane, nOutputPlane)
  }
}

private[bigdl] object Conv3DBackpropFilterV2 {
  def apply[T: ClassTag](
    dT: Int,
    dH: Int,
    dW: Int,
    padT: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): Conv3DBackpropFilterV2[T]
  = new Conv3DBackpropFilterV2[T](dT, dH, dW, padT, padH, padW, format)
}

private[bigdl] class Conv3DBackpropInput[T: ClassTag](
  dT: Int,
  dH: Int,
  dW: Int,
  padT: Int,
  padH: Int,
  padW: Int,
  format: DataFormat
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val fGradInput = Tensor[T]()

  protected def getInputSize(inputs: Table): Array[Int] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)

    if (format == DataFormat.NHWC) {
      val N = input.size(1)
      val D = input.size(2)
      val H = input.size(3)
      val W = input.size(4)
      val C = input.size(5)
      Array(N, C, D, H, W)
    } else {
      val N = input.size(1)
      val C = input.size(2)
      val D = input.size(3)
      val H = input.size(4)
      val W = input.size(5)
      Array(N, C, D, H, W)
    }
  }

  override def updateOutput(inputs: Table): Tensor[T] = {

    val filter: Tensor[T] = inputs[Tensor[T]](2)
    val outputBackprop: Tensor[T] = inputs[Tensor[T]](3)

    val transOutBackprop = if (format == DataFormat.NHWC) {
      // backpropInput only use input size, so we do not need it to be contiguous
      outputBackprop.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
    } else {
      outputBackprop
    }

    val transInputSize = getInputSize(inputs)

    val kT = filter.size(1)
    val kH = filter.size(2)
    val kW = filter.size(3)
    val nInputPlane = filter.size(4)
    val nOutputPlane = filter.size(5)

    var transWeight = filter.transpose(1, 5)
    transWeight = transWeight.transpose(2, 4)
    transWeight = transWeight.transpose(3, 5)
    transWeight = transWeight.contiguous()
    val weightMM = transWeight.view(nOutputPlane, nInputPlane * kT * kH * kW)

    VolumetricConvolution.conv3DBackpropInput(transInputSize, output, transOutBackprop,
      weightMM, fGradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH)

    if (format == DataFormat.NHWC) {
      output = output.transpose(2, 5)
      output = output.transpose(2, 3)
      output = output.transpose(3, 4)
      output = output.contiguous()
    }
    output
  }

  override def clearState(): Conv3DBackpropInput.this.type = {
    super.clearState()
    fGradInput.set()
    this
  }
}

private[bigdl] object Conv3DBackpropInput {
  def apply[T: ClassTag](
    dT: Int,
    dH: Int,
    dW: Int,
    padT: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): Conv3DBackpropInput[T]
  = new Conv3DBackpropInput[T](dT, dH, dW, padT, padH, padW, format)
}

private[bigdl] class Conv3DBackpropInputV2[T: ClassTag](dT: Int, dH: Int, dW: Int,
  padT: Int, padH: Int, padW: Int,
  format: DataFormat)
  (implicit ev: TensorNumeric[T])
  extends Conv3DBackpropInput[T](dT, dH, dW, padT, padH, padW, format) {

  private val fGradInput = Tensor[T]()

  override protected def getInputSize(inputs: Table): Array[Int] = {
    val inputSize: Tensor[Int] = inputs[Tensor[Int]](1)

    if (format == DataFormat.NHWC) {
      val N = inputSize.valueAt(1)
      val D = inputSize.valueAt(2)
      val H = inputSize.valueAt(3)
      val W = inputSize.valueAt(4)
      val C = inputSize.valueAt(5)
      Array(N, C, D, H, W)
    } else {
      val N = inputSize.valueAt(1)
      val C = inputSize.valueAt(2)
      val D = inputSize.valueAt(3)
      val H = inputSize.valueAt(4)
      val W = inputSize.valueAt(5)
      Array(N, C, D, H, W)
    }
  }

  override def clearState(): Conv3DBackpropInputV2.this.type = {
    super.clearState()
    fGradInput.set()
    this
  }
}

private[bigdl] object Conv3DBackpropInputV2 {
  def apply[T: ClassTag](
    dT: Int,
    dH: Int,
    dW: Int,
    padT: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): Conv3DBackpropInputV2[T]
  = new Conv3DBackpropInputV2[T](dT, dH, dW, padT, padH, padW, format)
}

private[bigdl] abstract class UnaryGrad[T: ClassTag, D: ClassTag](
  gradFirst: Boolean = false,
  needForward: Boolean = false)
  (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  type Module = AbstractModule[Tensor[D], Tensor[D], _]

  val module: Module

  override def updateOutput(input: Table): Tensor[D] = {
    val (grads, inputs) = if (gradFirst) {
      (input[Tensor[D]](1), input[Tensor[D]](2))
    } else {
      (input[Tensor[D]](2), input[Tensor[D]](1))
    }

    if (needForward) {
      module.forward(inputs)
    }

    output = module.updateGradInput(inputs, grads).toTensor[D]
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] class Relu6Grad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends UnaryGrad[T, D](true) {

  val module: Module = ReLU6Layer[D]()
}

private[bigdl] object Relu6Grad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Relu6Grad[T, D] =
    new Relu6Grad[T, D]()
}

private[bigdl] class EluGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends UnaryGrad[T, D](true, true) {

  override val module: Module = ELULayer[D]()
}

private[bigdl] object EluGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): EluGrad[T, D] = new EluGrad[T, D]()
}

private[bigdl] class TanhGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) extends Operation[Table, Tensor[D], T]{

  private val module = Tanh[D]()
  override def updateOutput(input: Table): Tensor[D] = {
    val (y, grads) = (input[Tensor[D]](1), input[Tensor[D]](2))

    output = module.updateGradInputInternal(y, grads).toTensor[D]
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array(scala.reflect.classTag[T], scala.reflect.classTag[D]), Array(ev, ev2))
  }
}

private[bigdl] object TanhGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): TanhGrad[T, D] =
    new TanhGrad[T, D]()
}

private[bigdl] class SoftplusGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends UnaryGrad[T, D](true, true) {

  override val module: Module = SoftPlusLayer[D]()

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] object SoftplusGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): SoftplusGrad[T, D] =
    new SoftplusGrad[T, D]()
}

private[bigdl] class SoftsignGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends UnaryGrad[T, D](true) {

  override val module: Module = SoftSignLayer[D]()
}

private[bigdl] object SoftsignGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): SoftsignGrad[T, D] =
    new SoftsignGrad[T, D]()
}

private[bigdl] class SigmoidGrad[T: ClassTag, D: ClassTag]
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  private val module = Sigmoid[D]()
  override def updateOutput(input: Table): Tensor[D] = {
    val (y, grads) = (input[Tensor[D]](1), input[Tensor[D]](2))

    output = module.updateGradInputInternal(y, grads).toTensor[D]
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] object SigmoidGrad {
  def apply[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): SigmoidGrad[T, D] =
    new SigmoidGrad[T, D]()
}

private[bigdl] class MaxPool[T: ClassTag](
  val ksize: Array[Int],
  val strides: Array[Int],
  val padding: String,
  val format: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], Tensor[T], T] {
  val pool: SpatialMaxPooling[T] = format match {
    case DataFormat.NHWC =>
      if (padding == "SAME") {
        SpatialMaxPooling(
          kH = ksize(1),
          kW = ksize(2),
          dH = strides(1),
          dW = strides(2),
          padH = -1,
          padW = -1,
          format = format
        )
      } else if (padding == "VALID") {
        SpatialMaxPooling(
          kH = ksize(1),
          kW = ksize(2),
          dH = strides(1),
          dW = strides(2),
          format = format
        )
      } else {
        throw new RuntimeException("Padding can only support SAME and VALID padding")
      }
    case DataFormat.NCHW =>
      if (padding == "SAME") {
        SpatialMaxPooling(
          kH = ksize(2),
          kW = ksize(3),
          dH = strides(2),
          dW = strides(3),
          padH = -1,
          padW = -1,
          format = format
        )
      } else if (padding == "VALID") {
        SpatialMaxPooling(
          kH = ksize(2),
          kW = ksize(3),
          dH = strides(2),
          dW = strides(3),
          format = format
        )
      } else {
        throw new RuntimeException("Padding can only support SAME and VALID padding")
      }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = pool.updateOutput(input)
    output
  }
}

private[bigdl] object MaxPool {
  def apply[T: ClassTag](
    ksize: Array[Int],
    strides: Array[Int],
    padding: String,
    format: DataFormat = DataFormat.NHWC
  )(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new MaxPool(ksize, strides, padding, format))
}

private[bigdl] class MaxPoolGrad[T: ClassTag](
  kH: Int,
  kW: Int,
  strideW: Int,
  strideH: Int,
  padH: Int,
  padW: Int,
  format: DataFormat
)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[T], T]{

  private var module : SpatialMaxPooling[T] = _

  override def updateOutput(input: Table): Tensor[T] = {
    if (module == null) {
      module = SpatialMaxPooling[T](
        kH,
        kW,
        strideH,
        strideW,
        padH,
        padW,
        format
      )
    }

    val inputData = input[Tensor[T]](1)
    val gradOutput = input[Tensor[T]](3)
    module.updateOutput(inputData)
    output = module.updateGradInput(inputData, gradOutput)
    output
  }
}

private[bigdl] object MaxPoolGrad {
  def apply[T: ClassTag](
    kH: Int,
    kW: Int,
    strideW: Int,
    strideH: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): MaxPoolGrad[T] =
    new MaxPoolGrad(kH, kW, strideW, strideH, padH, padW, format)
}

/**
 * LRNGrad calculate the backprop gradients of the Local response normalization layer.
 *
 * @param depthRadius
 * @param bias
 * @param alpha
 * @param beta
 * @param ev$1
 * @param ev
 * @param ev2
 * @tparam T Numeric type. Only support float/double now
 */
private[bigdl] class LRNGrad[T: ClassTag](
  depthRadius: Int = 5,
  bias: Float = 1.0f,
  alpha: Float = 1.0f,
  beta: Float = 0.5f
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[Float])
  extends Operation[Table, Tensor[Float], T] {

  output = Tensor[Float]()

  override def updateOutput(input: Table): Tensor[Float] = {
    val gradOutput = input[Tensor[Float]](1)
    val inputTensor = input[Tensor[Float]](2)
    val outputTensor = input[Tensor[Float]](3)

    output.resizeAs(inputTensor)
    var b = 1
    while(b <= inputTensor.size(1)) {
      SpatialCrossMapLRN.backwardFrameNHWCFloat(
        gradOutput.select(1, b),
        inputTensor.select(1, b),
        output.select(1, b),
        outputTensor.select(1, b),
        alpha * (2 * depthRadius + 1), 2 * depthRadius + 1, beta, bias
      )
      b += 1
    }
    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

private[bigdl] object LRNGrad {
  def apply[T: ClassTag](
    depthRadius: Int = 5,
    bias: Float = 1.0f,
    alpha: Float = 1.0f,
    beta: Float = 0.5f
  )(implicit ev: TensorNumeric[T]): LRNGrad[T]
  = new LRNGrad(depthRadius, bias, alpha, beta)
}

/**
 * This is similar to SpatialBatchNormalization.
 *
 * When isTraining is true, it takes three tensors as inputs, which is image,
 * scale, offset.
 *
 * The operation implemented is:
 *
 *         ( image - batch-mean(x) )
 * y = ---------------------------------- * weight + offset
 *      batch-standard-deviation(x)
 *
 * The operation will output y, mean and variance tensors.
 *
 * If the isTraining is false, it takes five tensors as inputs, which is image, scale, offset, mean,
 * and variance.
 *
 * @param epsilon
 * @param isTraining
 * @param momentum
 * @param dataFormat
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
private[bigdl] class FusedBatchNorm[T: ClassTag](
  epsilon: Float = 0.0001f,
  isTraining: Boolean = true,
  momentum: Float = 0.1f,
  dataFormat: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Table, T]{

  @transient
  private var runningMean: Tensor[Float] = null

  @transient
  private var runningVar: Tensor[Float] = null

  @transient
  private var saveStd: Tensor[Float] = null

  override def updateOutput(input: Table): Table = {
    val x = input[Tensor[Float]](1)
    val scale = input[Tensor[Float]](2)
    val offset = input[Tensor[Float]](3)
    val mean = input[Tensor[Float]](4)
    val variance = input[Tensor[Float]](5)

    if (output.length() == 0) {
      output(1) = Tensor[Float]().resizeAs(x) // y
      output(2) = Tensor[Float](x.size(4)) // batch mean
      output(3) = Tensor[Float](x.size(4)) // batch var
      output(4) = Tensor[Float](x.size(4)) // save mean
      output(5) = Tensor[Float](x.size(4)) // save var
      runningMean = Tensor[Float](x.size(4)) // running mean
      runningVar = Tensor[Float](x.size(4)) // running var
      saveStd = Tensor[Float](x.size(4)) // save std
    }

    val y = output[Tensor[Float]](1)
    val batchMean = output[Tensor[Float]](2)
    val batchVar = output[Tensor[Float]](3)
    val saveMean = output[Tensor[Float]](4)
    val saveVar = output[Tensor[Float]](5)

    if (isTraining) {
      if (dataFormat == DataFormat.NHWC) {
        SpatialBatchNormalization.updateOutputNHWCTrainFloat(
          x, y, batchMean, saveStd, runningMean, runningVar, scale, offset, epsilon, momentum,
          batchVar, saveVar
        )
      } else {
        SpatialBatchNormalization.updateOutputNCHWTrainFloat(
          x, y, batchMean, saveStd, runningMean, runningVar, scale, offset, epsilon, momentum,
          batchVar, saveVar
        )
      }
      saveMean.copy(batchMean)
    } else {
      if (dataFormat == DataFormat.NHWC) {
        SpatialBatchNormalization.updateOutputNHWCInferFloat(
          x, y, mean, variance, scale, offset, epsilon
        )
      } else {
        SpatialBatchNormalization.updateOutputNCHWInferFloat(
          x, y, mean, variance, scale, offset, epsilon
        )
      }
    }

    output
  }
}

private[bigdl] object FusedBatchNorm {
  def apply[T: ClassTag](epsilon: Float = 0.0001f, isTrainning: Boolean = true,
    momentum: Float = 0.1f, dataFormat: DataFormat = DataFormat.NHWC)
    (implicit ev: TensorNumeric[T]): FusedBatchNorm[T]
  = new FusedBatchNorm(epsilon, isTrainning, momentum, dataFormat)
}

/**
 * This is the gradient operation coressponding to the FusedBatchNorm. It will calculate the
 * activity, weight and bias gradients of the spatial batch normalization.
 *
 * The formula is
 *      x_backprop = scale * rsqrt(variance + epsilon) * [y_backprop - mean(y_backprop) -
 *     (x - mean(x)) * mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
 *     weight_backprop = sum(y_backprop * (x - mean(x)) * rsqrt(variance + epsilon))
 *     bias_backprop = sum(y_backprop)
 *
 * @param epsilon
 * @param dataFormat
 * @param isTrain
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
private[bigdl] class FusedBatchNormGrad[T: ClassTag](
  val epsilon: Float, val dataFormat: DataFormat,
  val isTrain: Boolean = false)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T]{


  private val gMean = Tensor[Float]()
  private val gxMean = Tensor[Float]()
  private val saveStd = Tensor[Float]()

  override def updateOutput(input: Table): Table = {
    val gradOutput = input[Tensor[Float]](1)
    val x = input[Tensor[Float]](2)
    val scale = input[Tensor[Float]](3)
    val saveMean = input[Tensor[Float]](4)
    val saveVar = input[Tensor[Float]](5)

    if (output.length() == 0) {
      output(1) = Tensor[Float]().resizeAs(x) // gradInput
      output(2) = Tensor[Float](x.size(4)) // weight gradient
      output(3) = Tensor[Float](x.size(4)) // bias gradient
      saveStd.resize(x.size(4)) // bias gradient
    }
    saveStd.copy(saveVar)
    saveStd.add(epsilon).pow(-0.5f)
    val gradInput = output[Tensor[Float]](1)
    val gradWeight = output[Tensor[Float]](2)
    val gradBias = output[Tensor[Float]](3)

    SpatialBatchNormalization.updateGradInputNHWCTrainFloat(
      x, gradOutput, gradInput, scale, saveMean, saveStd, gMean, gxMean)

    gradWeight.zero()
    gradBias.zero()
    SpatialBatchNormalization.accGradientNHWCFloat(
      gradOutput, gradWeight, gradBias, x, saveMean, saveStd, 1.0f, 1.0f)

    output
  }
}

private[bigdl] object FusedBatchNormGrad {
  def apply[T: ClassTag](epsilon: Float = 0.0001f, dataFormat: DataFormat = DataFormat.NHWC,
    isTraining: Boolean = true)(implicit ev: TensorNumeric[T]): FusedBatchNormGrad[T] =
    new FusedBatchNormGrad(epsilon, dataFormat, isTraining)
}

private[bigdl] class AvgPoolGrad[T: ClassTag](
  kH: Int,
  kW: Int,
  strideW: Int,
  strideH: Int,
  padH: Int,
  padW: Int,
  format: DataFormat
)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[T], T]{

  private var module : SpatialAveragePooling[T] = _

  override def updateOutput(input: Table): Tensor[T] = {
    if (module == null) {
      module = SpatialAveragePooling[T](
        kH,
        kW,
        strideH,
        strideW,
        padH,
        padW,
        countIncludePad = false,
        format = format
      )
    }

    val inputDataSize = input[Tensor[Int]](1).storage().array()

    val gradOutput = input[Tensor[T]](2)
    output = module.updateGradInputInternal(inputDataSize, gradOutput)
    output
  }
}

private[bigdl] object AvgPoolGrad {
  def apply[T: ClassTag](
    kH: Int,
    kW: Int,
    strideW: Int,
    strideH: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): AvgPoolGrad[T] =
    new AvgPoolGrad(kH, kW, strideW, strideH, padH, padW, format)
}

private[bigdl] class BiasAddGrad[T: ClassTag](dataFormat: DataFormat)
  (implicit ev: TensorNumeric[T])
  extends Operation[Tensor[T], Tensor[T], T] {

  private val module = BiasAdd()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    getBiasDims(input)
    output.resizeAs(input).copy(input)
    dataFormat match {
      case DataFormat.NCHW =>
        output = output.resize(Array(batch, channel, height, width)).sum(1)
        output = output.sum(3)
        output = output.sum(4)
      case DataFormat.NHWC =>
        output = output.resize(Array(batch * height * width, channel)).sum(1)
    }
    output
  }

  private var batch : Int = 1
  private var channel : Int = 1
  private var width : Int = 1
  private var height : Int = 1

  private def getBiasDims(tensor: Tensor[_]): Unit = {
    batch = 1
    channel = 1
    width = 1
    height = 1
    dataFormat match {
      case DataFormat.NHWC =>
        val channelDim = tensor.dim()
        channel = tensor.size(channelDim)
        var i = 1
        while(i < channelDim) {
          batch *= tensor.size(i)
          i += 1
        }
      case DataFormat.NCHW =>
        val channelDim = tensor.dim() - 2
        val heightDim = tensor.dim() - 1
        val widthDim = tensor.dim()
        channel = tensor.size(channelDim)
        height = tensor.size(heightDim)
        width = tensor.size(widthDim)
        var i = 1
        while(i < channelDim) {
          batch *= tensor.size(i)
          i += 1
        }
    }
  }
}

private[bigdl] object BiasAddGrad {
  def apply[T: ClassTag](dataFormat: DataFormat)
    (implicit ev: TensorNumeric[T]): BiasAddGrad[T] = new BiasAddGrad(dataFormat)
}

private[bigdl] class ReluGrad[T: ClassTag](implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[T], T]{

  val module = ReLULayer[T]()

  override def updateOutput(input: Table): Tensor[T] = {
    val grads = input[Tensor[T]](1)
    val inputs = input[Tensor[T]](2)

    output = module.updateGradInput(inputs, grads).toTensor[T]
    output
  }
}

private[bigdl] object ReluGrad {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): ReluGrad[T] = new ReluGrad()
}
