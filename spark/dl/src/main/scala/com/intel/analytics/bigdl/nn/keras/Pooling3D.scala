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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Abstract class for different pooling 3D layers.
 * Do not create a new instance of it or use it in a model.
 * Please use its child classes, 'AveragePooling3D' and 'MaxPooling3D' instead.
 */
abstract class Pooling3D[T: ClassTag](
   val poolSize: Array[Int] = Array(2, 2, 2),
   val strides: Array[Int] = null,
   val dimOrdering: String = "CHANNEL_FIRST",
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering.toLowerCase() == "channel_first", s"Pooling3D currently only supports " +
    s"format CHANNEL_FIRST, but got format $dimOrdering")

  require(poolSize.length == 3,
    s"For Pooling3D, poolSize should be of length 3 but got length ${poolSize.length}")

  val strideValues: Array[Int] = if (strides == null) poolSize else strides
  require(strideValues.length == 3,
    s"For Pooling3D, strides should be of length 3 but got length ${strideValues.length}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"Pooling3D requires 5D input, but got input dim ${input.length}")
    val dim1Length = KerasUtils.computeConvOutputLength(input(2), poolSize(0),
      "valid", strideValues(0))
    val dim2Length = KerasUtils.computeConvOutputLength(input(3), poolSize(1),
      "valid", strideValues(1))
    val dim3Length = KerasUtils.computeConvOutputLength(input(4), poolSize(2),
      "valid", strideValues(2))
    Shape(input(0), input(1), dim1Length, dim2Length, dim3Length)
  }
}
