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
import com.intel.analytics.bigdl.nn.{Transpose, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Just a wrapper class. Please use Activation('softmax') instead.
 */
class SoftMax[T: ClassTag](
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 2 || input.length == 3 || input.length == 4,
      s"SoftMax requires 2D or 3D or 4D input, but got input dim ${input.length}")
    inputShape
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = com.intel.analytics.bigdl.nn.SoftMax()
    if (input.length <= 2) {
      layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    } else if (input.length == 3) {
      val model = TSequential[T]()
      model.add(Transpose(Array((1, 3))))
      model.add(layer)
      model.add(Transpose(Array((1, 3))))
      model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    } else if (input.length == 4) {
      val model = TSequential[T]()
      model.add(Transpose(Array((2, 4))))
      model.add(layer)
      model.add(Transpose(Array((2, 4))))
      model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    } else {
      throw new IllegalArgumentException(s"SoftMax requires 2D or 3D or 4D input, " +
        s"but got input dim ${input.length}")
    }
  }
}

object SoftMax {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T](inputShape)
  }
}
