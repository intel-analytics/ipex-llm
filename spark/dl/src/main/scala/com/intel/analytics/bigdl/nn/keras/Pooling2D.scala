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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

abstract class Pooling2D[T: ClassTag](
   val poolSize: (Int, Int) = (2, 2),
   var strides: (Int, Int) = null,
   val borderMode: String = "valid",
   val format: DataFormat = DataFormat.NCHW,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
    s"Pooling2D: $borderMode")

  def strideValues: (Int, Int) = if (strides == null) poolSize else strides

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"Pooling2D requires 4D input, but got input dim ${input.length}")
    val (dimH, dimW, dimC) = format.getHWCDims(4)
    val rows = KerasUtils.computeConvOutputLength(input(dimH -1), poolSize._1,
      borderMode, strideValues._1)
    val cols = KerasUtils.computeConvOutputLength(input(dimW -1), poolSize._2,
      borderMode, strideValues._2)
    format match {
      case DataFormat.NCHW => Shape(input(0), input(1), rows, cols)
      case DataFormat.NHWC => Shape(input(0), rows, cols, input(3))
    }
  }

}
