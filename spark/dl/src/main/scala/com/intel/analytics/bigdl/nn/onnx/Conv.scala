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


package com.intel.analytics.bigdl.nn.onnx

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


case class Conv[T: ClassTag](
  nInputPlane: Int, // BigDL requires
  nOutputPlane: Int, // BigDL requires
  kernelShape: List[Int],
  weight: Tensor[T], // BigDL requires
  bias: Tensor[T], // BigDL requires
  autoPad: String, // missing in BigDL
  dilations: List[Int],
  group: Int,
  pads: List[Int],
  strides: List[Int]
)


object Conv {
  def apply[T: ClassTag](
    nInputPlane: Int, // BigDL requires
    nOutputPlane: Int, // BigDL requires
    kernelShape: List[Int],
    weight: Tensor[T], // BigDL requires
    bias: Tensor[T], // BigDL requires
    autoPad: String = "NOTSET", // missing in BigDL
    dilations: List[Int] = null,
    group: Int = 1,
    pads: List[Int] = null,
    strides: List[Int] = null
  )(implicit ev: TensorNumeric[T]): nn.SpatialConvolution[T] = {

    val (dilationW: Int, dilationH: Int) = dilations match {
      case null => (1, 1)
      // case List(weight, height) => (weight, height)
      case _ => throw new IllegalArgumentException("dilation")
    }

    val (kW: Int, kH: Int) = kernelShape match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException()
    }
    val (dW: Int, dH: Int) = strides match {
      case null => (1, 1)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException()
    }
    val (padW: Int, padH: Int) = pads match {
      case null => (0, 0)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException()
    }

    val conv = new nn.SpatialConvolution(
      nInputPlane = nInputPlane,
      nOutputPlane = nOutputPlane,
      kernelW = kW, kernelH = kH,
      strideW = dW, strideH = dH,
      nGroup = group)

    conv.setWeightsBias(Array(weight, bias))

    conv
  }
}
