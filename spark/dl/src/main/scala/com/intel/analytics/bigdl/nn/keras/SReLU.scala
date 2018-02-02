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

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class SReLU[T: ClassTag](SharedAxes: Array[Int] = null,
                         var inputShape: Shape = null
  )(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val shape = inputShape.toSingle().toArray
    if (SharedAxes == null) {
      val layer = com.intel.analytics.bigdl.nn.SReLU(shape.slice(1, shape.length))
      layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    } else
    {
      val layer = com.intel.analytics.bigdl.nn.SReLU(shape.slice(1, shape.length), SharedAxes)
      layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
  }
}


object SReLU {

  def apply[@specialized(Float, Double) T: ClassTag](
    SharedAxes: Array[Int] = null,
    inputShape: Shape = null
    )(implicit ev: TensorNumeric[T]) : SReLU[T] = {
    new SReLU[T](
      SharedAxes,
      inputShape)
  }
}
