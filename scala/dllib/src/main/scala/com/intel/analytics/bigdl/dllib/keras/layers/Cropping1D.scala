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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.nn.keras.{Cropping1D => BigDLCropping1D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

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
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Cropping1D[T: ClassTag](
    override val cropping: Array[Int] = Array(1, 1),
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLCropping1D[T](
    cropping, inputShape) with Net {
}

object Cropping1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: (Int, Int) = (1, 1),
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping1D[T] = {
    val croppingArray = cropping match {
      case null => throw new IllegalArgumentException("For Cropping1D, " +
        "cropping can not be null, please input int tuple of length 2")
      case _ => Array(cropping._1, cropping._2)
    }
    new Cropping1D[T](croppingArray, inputShape)
  }
}
