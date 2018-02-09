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

/**
 * Abstract class for different pooling 2D layers.
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'AveragePooling2D' and 'MaxPooling2D' instead.
 */
abstract class Pooling2D[T: ClassTag](
   val poolSize: Array[Int] = Array(2, 2),
   val strides: Array[Int] = null,
   val borderMode: String = "valid",
   val dimOrdering: DataFormat = DataFormat.NCHW,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(poolSize.length == 2,
    s"For Pooling2D, poolSize should be of length 2 but got length ${poolSize.length}")
  require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
    s"Pooling2D: $borderMode")

  val strideValues: Array[Int] = if (strides == null) poolSize else strides
  require(strideValues.length == 2,
    s"For Pooling2D, strides should be of length 2 but got length ${strideValues.length}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"Pooling2D requires 4D input, but got input dim ${input.length}")
    val (dimH, dimW, dimC) = dimOrdering.getHWCDims(4)
    val rows = KerasUtils.computeConvOutputLength(input(dimH -1), poolSize(0),
      borderMode, strideValues(0))
    val cols = KerasUtils.computeConvOutputLength(input(dimW -1), poolSize(1),
      borderMode, strideValues(1))
    dimOrdering match {
      case DataFormat.NCHW => Shape(input(0), input(1), rows, cols)
      case DataFormat.NHWC => Shape(input(0), rows, cols, input(3))
    }
  }
}
