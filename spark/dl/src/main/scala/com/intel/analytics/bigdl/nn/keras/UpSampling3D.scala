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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class UpSampling3D[T: ClassTag](
   val size: (Int, Int, Int) = (2, 2, 2),
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.UpSampling3D(Array(size._1, size._2, size._3))
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object UpSampling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: (Int, Int, Int) = (2, 2, 2),
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): UpSampling3D[T] = {
    new UpSampling3D[T](size, inputShape)
  }
}
