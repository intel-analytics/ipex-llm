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

abstract class Pooling3D[T: ClassTag](
   val poolSize: (Int, Int, Int) = (2, 2, 2),
   val strides: (Int, Int, Int) = null,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  def strideValues: (Int, Int, Int) = if (strides == null) poolSize else strides

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"Pooling3D requires 5D input, but got input dim ${input.length}")
    val dim1Length = KerasUtils.computeConvOutputLength(input(2), poolSize._1,
      "valid", strideValues._1)
    val dim2Length = KerasUtils.computeConvOutputLength(input(3), poolSize._2,
      "valid", strideValues._2)
    val dim3Length = KerasUtils.computeConvOutputLength(input(4), poolSize._3,
      "valid", strideValues._3)
    Shape(input(0), input(1), dim1Length, dim2Length, dim3Length)
  }

}
