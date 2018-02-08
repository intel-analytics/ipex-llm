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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class TimeDistributed[T: ClassTag](
   val layer: KerasLayer[Tensor[T], Tensor[T], T],
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  private def getInnerInput(input: Array[Int]): Array[Int] = {
    Array(input(0)) ++ input.slice(2, input.length)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length >=3,
      s"TimeDistributed requires at least 3D input, but got input dim ${input.length}")
    val innerInput = getInnerInput(input)
    val innerOutput = layer.build(Shape(innerInput)).toSingle()
    val output = innerOutput.take(1) ++ List(input(1)) ++ innerOutput.drop(1)
    Shape(output.toArray)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val innerInput = getInnerInput(input)
    val klayer = layer.doBuild(Shape(innerInput)).asInstanceOf[TensorModule[T]]
    val timedistributed = com.intel.analytics.bigdl.nn.TimeDistributed(klayer)
    timedistributed.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object TimeDistributed {
    def apply[@specialized(Float, Double) T: ClassTag](
    layer: KerasLayer[Tensor[T], Tensor[T], T],
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T](layer, inputShape)
  }
}
