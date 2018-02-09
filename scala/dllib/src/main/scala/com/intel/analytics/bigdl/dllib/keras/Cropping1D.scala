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

import com.intel.analytics.bigdl.nn.SpatialZeroPadding
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Cropping layer for 1D input (e.g. temporal sequence).
 * The input of this layer should be 3D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param cropping Int array of length 2. How many units should be trimmed off
 *                 at the beginning and end of the cropping dimension. Default is (1, 1).
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Cropping1D[T: ClassTag](
   val cropping: Array[Int] = Array(1, 1),
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(cropping.length == 2,
    s"For Cropping1D, cropping values should be of length 2 but got length ${cropping.length}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 3,
      s"Cropping1D requires 3D input, but got input dim ${input.length}")
    Shape(input(0), input(1)-cropping(0)-cropping(1), input(2))
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = SpatialZeroPadding(0, 0, -cropping(0), -cropping(1))
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Cropping1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: (Int, Int) = (1, 1),
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping1D[T] = {
    new Cropping1D[T](Array(cropping._1, cropping._2), inputShape)
  }
}
