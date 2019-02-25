/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule}
import com.intel.analytics.bigdl.nn.keras.{SoftMax => KSoftMax}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{Transpose, Sequential => TSequential,
TimeDistributed => TTimeDistributed}

import scala.reflect.ClassTag

class SoftMax[T: ClassTag](override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KSoftMax[T](inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    require(input.length < 5, "SoftMax only support 2d/3d/4d input")
    val layer = com.intel.analytics.bigdl.nn.SoftMax()
    if (input.length <= 2) {
      layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
    else {
      val seq = TSequential[T]()
      seq.add(Transpose(Array((1, 3))))
      seq.add(layer)
      seq.add(Transpose(Array((1, 3))))

      val model = if (input.length > 3) {
        TTimeDistributed[T](seq.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]])
      }
      else seq

      model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
  }
}

object SoftMax {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T](inputShape)
  }
}
