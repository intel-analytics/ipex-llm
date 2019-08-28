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


case class MaxPool[T: ClassTag] (
  autoPad: String,
  ceilMode: Int,
  dilations: List[Int],
  kernelShape: List[Int],
  pads: List[Int],
  storageOrder: Int,
  strides: List[Int]
)

object MaxPool {
  def apply[T: ClassTag](
    autoPad: String = "NOTSET",
    ceilMode: Int = 0,
    dilations: List[Int] = List(),
    kernelShape: List[Int],
    pads: List[Int] = List(),
    storageOrder: Int = 0,
    strides: List[Int] = List()
  )(implicit ev: TensorNumeric[T]): nn.SpatialMaxPooling[T] = {

    val (kW: Int, kH: Int) = kernelShape match {
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException()
    }
    val (dW: Int, dH: Int) = strides match {
      case List() => (1, 1)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException()
    }
    val (padW: Int, padH: Int) = pads match {
      case List() => (0, 0)
      case List(width, height) => (width, height)
      case _ => throw new IllegalArgumentException()
    }

    if (ceilMode != 0) {
      throw new IllegalArgumentException("MaxPool doesnt support ceil mode yet")
    }

    if (storageOrder != 0) {
      throw new IllegalArgumentException("MaxPool doesnt support storage order yet")
    }

    new nn.SpatialMaxPooling(kW = kW, kH = kH, dW = dW, dH = dH, padW = padW, padH = padH)

  }

}
