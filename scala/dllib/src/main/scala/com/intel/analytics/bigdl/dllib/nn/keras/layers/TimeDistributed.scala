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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.utils.SingleShape
import com.intel.analytics.bigdl.nn.keras.{KerasLayer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalTimeDistributed
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet

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
  val layer: KerasLayer[Activity, Tensor[T], T],
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Tensor[T], T](KerasUtils.addBatch(inputShape)) with Net {

  private var seqLen: Int = 0

  private def getInnerShape(inputShape: Shape): Shape = {
    val sizes = inputShape.toSingle().toArray
    require(sizes.length >= 3,
      s"TimeDistributed requires at least 3D input, but got input dim ${sizes.length}")
    if (seqLen != 0) {
      // in case time dim is singleton
      if (sizes(1) != 1) seqLen = sizes(1)
    } else seqLen = sizes(1)
    Shape(Array(sizes(0)) ++ sizes.drop(2))
  }

  private def getInnerOutputShape(shape: Shape): Shape = {
    val sizes = shape.toSingle().toArray
    Shape(Array(sizes(0), seqLen) ++ sizes.drop(1))
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val innerShape = if (inputShape.isInstanceOf[SingleShape]) getInnerShape(inputShape)
    else {
      val shapes = inputShape.toMulti()
      Shape(shapes.map(getInnerShape(_)))
    }

    val innerOutputShape = layer.computeOutputShape(innerShape)
    getInnerOutputShape(innerOutputShape)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Tensor[T], T] = {
    val innerShape = if (inputShape.isInstanceOf[SingleShape]) getInnerShape(inputShape)
    else Shape(inputShape.toMulti().map(getInnerShape(_)))
    layer.build(innerShape)
    layer.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
    val timedistributed = InternalTimeDistributed[T](layer)
    timedistributed.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object TimeDistributed {
  def apply[@specialized(Float, Double) T: ClassTag](
      layer: KerasLayer[Activity, Tensor[T], T],
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T](layer, inputShape)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
      layer: ZooModel[Activity, Activity, T],
      inputShape: Shape)(implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    layer.model match {
      case keras: KerasNet[T] =>
        new TimeDistributed[T](keras.asInstanceOf[KerasLayer[Activity, Tensor[T], T]], inputShape)
      case _ => throw new Exception(s"$layer is not defined in Keras style")
    }
  }
}

