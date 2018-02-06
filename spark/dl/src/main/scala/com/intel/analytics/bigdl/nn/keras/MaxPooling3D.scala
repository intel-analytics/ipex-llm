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

import com.intel.analytics.bigdl.nn.VolumetricMaxPooling
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class MaxPooling3D[T: ClassTag](
   poolSize: Array[Int] = Array(2, 2, 2),
   strides: Array[Int] = null,
   inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Pooling3D[T](poolSize, strides, inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = VolumetricMaxPooling(
      kT = poolSize(0),
      kW = poolSize(2),
      kH = poolSize(1),
      dT = strideValues(0),
      dW = strideValues(2),
      dH = strideValues(1))
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object MaxPooling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: Array[Int] = Array(2, 2, 2),
    strides: Array[Int] = null,
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): MaxPooling3D[T] = {
    new MaxPooling3D[T](poolSize, strides, inputShape)
  }
}
