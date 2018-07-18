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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, TimeDistributed => BTimeDistributed}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalTimeDistributed
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * TimeDistributed wrapper.
 * Apply a layer to every temporal slice of an input.
 * The input should be at least 3D, and the dimension of index one
 * will be considered to be the temporal dimension.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * If you apply TimeDistributed to a Dense layer, you can use:
 * TimeDistributed(Dense(8), inputShape = Shape(10, 12))
 *
 * @param layer A layer instance.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class TimeDistributed[T: ClassTag](
    val layer: KerasLayer[Tensor[T], Tensor[T], T],
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  private def getInnerInput(input: Array[Int]): Array[Int] = {
    Array(input(0)) ++ input.slice(2, input.length)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length >= 3,
      s"TimeDistributed requires at least 3D input, but got input dim ${input.length}")
    val innerInput = getInnerInput(input)
    val innerOutput = layer.computeOutputShape(Shape(innerInput)).toSingle()
    val batch = innerOutput.take(1)
    val steps = List(input(1))
    val output = batch ++ steps ++ innerOutput.drop(1)
    Shape(output.toArray)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val innerInput = getInnerInput(input)
    layer.build(Shape(innerInput))
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    val timedistributed = InternalTimeDistributed[T](layer)
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

