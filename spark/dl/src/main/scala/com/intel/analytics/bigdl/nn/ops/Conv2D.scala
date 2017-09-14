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
  strides: Array[Int],
  padding: String,
  format: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private var conv: SpatialConvolution[T] = _

  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)

    conv = format match {
      case DataFormat.NHWC =>
        if (padding == "SAME") {
          SpatialConvolution(
            nInputPlane = input.size(4),
            nOutputPlane = filter.size(4),
            kernelH = filter.size(1),
            kernelW = filter.size(2),
            strideH = strides(1),
            strideW = strides(2),
            padH = -1,
            padW = -1,
            withBias = false,
            format = format
          )
        } else if (padding == "VALID") {
          SpatialConvolution(
            nInputPlane = input.size(4),
            nOutputPlane = filter.size(4),
            kernelH = filter.size(1),
            kernelW = filter.size(2),
            strideH = strides(1),
            strideW = strides(2),
            withBias = false,
            format = format
          )
        } else {
          throw new RuntimeException("Padding can only support SAME and VALID padding")
        }

      case DataFormat.NCHW =>
        if (padding == "SAME") {
          SpatialConvolution(
            nInputPlane = input.size(2),
            nOutputPlane = filter.size(4),
            kernelH = filter.size(1),
            kernelW = filter.size(2),
            strideH = strides(2),
            strideW = strides(3),
            padH = -1,
            padW = -1,
            withBias = false,
            format = format
          )
        } else if (padding == "VALID") {
          SpatialConvolution(
            nInputPlane = input.size(2),
            nOutputPlane = filter.size(4),
            kernelH = filter.size(1),
            kernelW = filter.size(2),
            strideH = strides(2),
            strideW = strides(3),
            withBias = false,
            format = format
          )
        } else {
          throw new RuntimeException("Padding can only support SAME and VALID padding")
        }
    }

    conv.setWeightsBias(Array(filter))
    output = conv.updateOutput(input)
    output
  }
}

object Conv2D {
  def apply[T: ClassTag](
    strides: Array[Int],
    padding: String,
    format: DataFormat = DataFormat.NHWC
  )(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new Conv2D(strides, padding, format))
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
    val sizes = input.toTable.apply[Tensor[Float]](1).squeeze()
    val kernel = input.toTable.apply[Tensor[T]](2)
    val data = input.toTable.apply[Tensor[T]](3)

    require(data.nDimension() == 4, s"Need a 4D input but is ${data.nDimension()}")
    require(sizes.nDimension() == 1, s"Need a 1D size but is ${sizes.nDimension()}")

    val (nOutputPlane, nInputPlane) = if (format == DataFormat.NCHW) {
      (data.size(2), sizes.valueAt(2).toInt)
    } else {
      (data.size(4), sizes.valueAt(4).toInt)
    }

    if (module == null) {
      module = new SpatialConvolution[T](
        nInputPlane = nInputPlane,
        nOutputPlane = nOutputPlane,
        kernelW = kernel.size(2),
        kernelH = kernel.size(1),
        strideH = strideH,
        strideW = strideW,
        padH = padH,
        padW = padW,
        initWeight = kernel,
        format = format,
        withBias = false
      )

      dummyInput = Tensor[T](sizes.valueAt(1).toInt, sizes.valueAt(2).toInt,
        sizes.valueAt(3).toInt, sizes.valueAt(4).toInt)
      module.forward(dummyInput)
    } else {
      val (nOutputPlanbe, nInputPlane) = if (format == DataFormat.NCHW) {
        (data.size(2), sizes.valueAt(2))
      } else {
        (data.size(4), sizes.valueAt(4))
      }

      require(module.nInputPlane == nInputPlane, "nInputPlane is not valid")
      require(module.nOutputPlane == nOutputPlane, "nOutputPlane is not valid")
      require(module.kernelH == kernel.size(1), "kernelH is not valid")
      require(module.kernelW == kernel.size(2), "kernelW is not valid")
      require(kernel.size(3) == nInputPlane, "kernel nInputPlane is not valid")
      require(kernel.size(4) == nOutputPlane, "kernel nOutputPlane is not valid")
      require(dummyInput.size(1) == sizes.valueAt(1), "size 1 is not correct")
      require(dummyInput.size(2) == sizes.valueAt(2), "size 1 is not correct")
      require(dummyInput.size(3) == sizes.valueAt(3), "size 1 is not correct")
      require(dummyInput.size(4) == sizes.valueAt(4), "size 1 is not correct")
    }

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

