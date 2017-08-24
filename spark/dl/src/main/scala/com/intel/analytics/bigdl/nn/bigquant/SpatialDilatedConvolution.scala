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

package com.intel.analytics.bigdl.nn.bigquant

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import scala.reflect.ClassTag

@SerialVersionUID(- 8572055756810843156L)
class SpatialDilatedConvolution[T: ClassTag](
  nInputPlane: Int, // The number of expected input planes in the image given into forward()
  nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  kernelW: Int, // The kernel width of the convolution
  kernelH: Int, // The kernel height of the convolution
  strideW: Int = 1, // The step of the convolution in the width dimension.
  strideH: Int = 1, // The step of the convolution in the height dimension
  padW: Int = 0, // The additional zeros added per width to the input planes.
  padH: Int = 0, // The additional zeros added per height to the input planes.
  val dilationW: Int = 1,
  val dilationH: Int = 1,
  nGroup: Int = 1 // Kernel group number
)(implicit ev: TensorNumeric[T]) extends SpatialConvolution[T](
  nInputPlane,
  nOutputPlane,
  kernelW,
  kernelH,
  strideW,
  strideH,
  padW,
  padH,
  nGroup
) {
  override val DILATION_WIDTH: Int = dilationW
  override val DILATION_HEIGHT: Int = dilationH

  override def toString(): String = {
    s"fixpoint.SpatialDilatedConvolution($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH, $dilationW, $dilationH)"
  }
}

object SpatialDilatedConvolution {
  def apply[@specialized(Float) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    dilationW: Int = 1,
    dilationH: Int = 1
  )(implicit ev: TensorNumeric[T]) : SpatialDilatedConvolution[T] = {
    new SpatialDilatedConvolution[T](nInputPlane, nOutputPlane, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH)
  }
}

