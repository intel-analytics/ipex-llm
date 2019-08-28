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


case class AveragePool[T: ClassTag](
   kernelShape: List[Int],
   autoPad: String,
   ceilMode: Int,
   countIncludePad: Int,
   pads: List[Int],
   strides: List[Int]
)


object AveragePool {
  def apply[T: ClassTag](
    kernelShape: List[Int],
    autoPad: String = "NOTSET",
    ceilMode: Int = 0,
    countIncludePad: Int = 0,
    pads: List[Int] = null,
    strides: List[Int] = null
  )(implicit ev: TensorNumeric[T]): nn.SpatialAveragePooling[T] = {
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

    new nn.SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH,
      ceilMode = if (ceilMode == 0) false else true,
      countIncludePad = if (countIncludePad == 0) false else true)
  }
}
