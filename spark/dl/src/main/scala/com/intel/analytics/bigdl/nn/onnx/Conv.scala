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


/**
 * The convolution operator consumes an input tensor, and computes the output.
 *
 * auto_pad : string (default is NOTSET)
 *  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
 *  Where default value is NOTSET, which means explicit padding is used.
 *  SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.
 *  In case of odd number add extra padding at the end for SAME_UPPER and the start for SAME_LOWER.
 *  VALID mean no padding.
 * dilations : list of ints. Dilation value along each spatial axis of the filter.
 * group : int (default is 1), number of groups input channels and output channels are divided into.
 * kernel_shape : list of ints.The shape of the convolution kernel.
 * pads: Padding for the beginning and ending along each spatial axis,
 * strides: Stride along each spatial axis.
 */
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
      case List(width: Int, height: Int) => (width.toInt, height.toInt)
      case _ => throw new IllegalArgumentException(
        "Dilations is expected in the form of List(width, height)," +
        "the input dilations: " + dilations)
    }

    val (kW: Int, kH: Int) = kernelShape match {
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Kernel shape is expected in the form of List(width, height)," +
        "the input kernel shape: " + kernelShape)
    }

    val (dW: Int, dH: Int) = strides match {
      case null => (1, 1)
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Strides is expected in the form of List(width, height)," +
        "the input strides: " + strides)
    }

    val (padW: Int, padH: Int) = pads match {
      case null => (0, 0)
      case List(width: Int, height: Int) => (width, height)
      case _ => throw new IllegalArgumentException(
        "Pads is expected in the form of List(width, height)," +
        "the input pads: " + strides)
    }


    if (dilationH != 1 && dilationW != 1) {
      throw new UnsupportedOperationException(
        "Dilations is expected to be (1, 1)" +
        "the input dilations: " + (dilationW, dilationH))
    }


    val conv = new nn.SpatialConvolution(
      nInputPlane = nInputPlane,
      nOutputPlane = nOutputPlane,
      kernelW = kW, kernelH = kH,
      strideW = dW, strideH = dH,
      padW = padW, padH = padH, nGroup = group,
      initWeight = weight, initBias = bias)

    conv
  }
}
