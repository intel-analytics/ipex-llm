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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class MaxPooling2D[T: ClassTag] (
   val poolSize: (Int, Int) = (2, 2),
   var strides: (Int, Int) = null,
   val borderMode: String = "valid",
   val format: DataFormat = DataFormat.NCHW,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
    s"MaxPooling2D: $borderMode")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val pads = KerasUtils.getPadsFromBorderMode(borderMode)
    if (strides == null) {
      strides = poolSize
    }
    val layer = SpatialMaxPooling(
      kW = poolSize._2,
      kH = poolSize._1,
      dW = strides._2,
      dH = strides._1,
      padW = pads._2,
      padH = pads._1,
      format = format
    )
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object MaxPooling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: (Int, Int) = (2, 2),
    strides: (Int, Int) = null,
    borderMode: String = "valid",
    format: DataFormat = DataFormat.NCHW,
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): MaxPooling2D[T] = {
    new MaxPooling2D[T](poolSize, strides, borderMode, format, inputShape)
  }
}
