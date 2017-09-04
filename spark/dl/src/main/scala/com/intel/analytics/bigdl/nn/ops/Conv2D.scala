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
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Conv2D[T: ClassTag](
  input: Tensor[T],
  filter: Tensor[T],
  strides: Array[Int],
  padding: String,
  format: DataFormat = DataFormat.NHWC
)(implicit ev: TensorNumeric[T]) extends Operation[Tensor[T], T] {

  val conv: SpatialConvolution[T] = format match {
    case DataFormat.NHWC =>
      if (padding == "SAME") {
        SpatialConvolution(
          nInputPlane = input.size(4),
          nOutputPlane = filter.size(4),
          kernelH = filter.size(2),
          kernelW = filter.size(3),
          strideH = strides(2),
          strideW = strides(3),
          padH = (filter.size(2) - 1) / 2,
          padW = (filter.size(3) - 1) / 2,
          format = format
        )
      } else if (padding == "VALID") {
        SpatialConvolution(
          nInputPlane = input.size(4),
          nOutputPlane = filter.size(4),
          kernelH = filter.size(2),
          kernelW = filter.size(3),
          strideH = strides(2),
          strideW = strides(3),
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
          kernelH = filter.size(3),
          kernelW = filter.size(4),
          strideH = strides(3),
          strideW = strides(4),
          padH = (filter.size(3) - 1) / 2,
          padW = (filter.size(4) - 1) / 2,
          format = format
        )
      } else if (padding == "VALID") {
        SpatialConvolution(
          nInputPlane = input.size(2),
          nOutputPlane = filter.size(4),
          kernelH = filter.size(3),
          kernelW = filter.size(4),
          strideH = strides(3),
          strideW = strides(4),
          format = format
        )
      } else {
        throw new RuntimeException("Padding can only support SAME and VALID padding")
      }
  }

  conv.weight.resizeAs(filter).copy(filter)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    conv.updateOutput(input)
  }
}

object Conv2D {
  def apply[T: ClassTag](
    input: Tensor[T],
    filter: Tensor[T],
    strides: Array[Int],
    padding: String,
    format: DataFormat = DataFormat.NHWC
  )(implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
  = ModuleToOperation[Tensor[T], T](new Conv2D(input, filter, strides, padding, format))
}
