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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.SpatialConvolution
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Conv2D[T: ClassTag](
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

object Conv2D {
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
class Conv2DTranspose[T: ClassTag](
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

object Conv2DTranspose {
  def apply[T: ClassTag](
    strideW: Int,
    strideH: Int,
    padW: Int = -1,
    padH: Int = -1,
    format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): Conv2DTranspose[T] =
    new Conv2DTranspose(strideW, strideH, padW, padH, format)
}

class Conv2DBackFilter[T: ClassTag](
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

object Conv2DBackFilter {
  def apply[T: ClassTag](
    strideW: Int,
    strideH: Int,
    padW: Int = -1,
    padH: Int = -1,
    format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): Conv2DBackFilter[T] =
    new Conv2DBackFilter(strideW, strideH, padW, padH, format)
}

