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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


/**
 * AveragePool consumes an input tensor X and applies average pooling across the tensor
 * according to kernel sizes, stride sizes, and pad lengths. average pooling consisting of
 * computing the average on all values of a subset of the input tensor according to the
 * kernel size and downsampling the data into the output tensor Y for further processing.
 *
 * https://github.com/onnx/onnx/blob/master/docs/Operators.md#averagepool
 *
 * kernelShape The size of the kernel along each axis. (required)
 * autoPad Currently must be NOTSET.
 *                auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
 *                Where default value is NOTSET, which means explicit padding is used.
 *                SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial
 *                size match the input.In case of odd number add the extra padding at the
 *                end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding.
 * ceilMode Wether to use ceil or floor (default) to compute the output shape. (default is 0)
 * countIncludePad Whether include pad pixels when calculating values for the edges.
 *                        Default is 0, doesn't count include pad.
 * pads Padding, the padding defaults to 0 along start and end of each spatial axis.
 * strides Stride along each spatial axis.
 */
object AveragePool {
  def apply[T: ClassTag](kernelShape: List[Int],
    autoPad: String = "NOTSET",
    ceilMode: Int = 0, countIncludePad: Int = 0,
    pads: List[Int] = null, strides: List[Int] = null
  )(implicit ev: TensorNumeric[T]): nn.SpatialAveragePooling[T] = {
    val (kW: Int, kH: Int) = kernelShape match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException("Bad kernel value: " + kernelShape)
    }
    val (dW: Int, dH: Int) = strides match {
      case null => (1, 1)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException("Bad strides value: " + strides)
    }
    val (padW: Int, padH: Int) = pads match {
      case null => (0, 0)
      case width :: height :: _ => (width, height)
      case _ => throw new IllegalArgumentException("Bad pads value: " + pads)
    }
    new nn.SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH,
      ceilMode = if (ceilMode == 0) false else true,
      countIncludePad = if (countIncludePad == 0) false else true)
  }
}
