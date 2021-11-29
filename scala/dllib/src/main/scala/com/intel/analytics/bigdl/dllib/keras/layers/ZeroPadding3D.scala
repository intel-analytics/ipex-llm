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

import com.intel.analytics.bigdl.dllib.nn.keras.{ZeroPadding3D => BigDLZeroPadding3D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Zero-padding layer for 3D data (spatial or spatio-temporal).
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param padding Int array of length 3.
 *                How many zeros to add at the beginning and end of the 3 padding dimensions.
 *                Symmetric padding will be applied to each dimension. Default is (1, 1, 1).
 * @param dimOrdering Format of the input data. Either "CHANNEL_FIRST" (dimOrdering='th') or
 *                    "CHANNEL_LAST" (dimOrdering='tf'). Default is "CHANNEL_FIRST".
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ZeroPadding3D[T: ClassTag](
   override val padding: Array[Int] = Array(1, 1, 1),
   override val dimOrdering: String = "CHANNEL_FIRST",
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLZeroPadding3D[T](padding, dimOrdering, inputShape) with Net {}

object ZeroPadding3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    padding: (Int, Int, Int) = (1, 1, 1),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : ZeroPadding3D[T] = {
    val paddingArray = padding match {
      case null => throw new IllegalArgumentException("For ZeroPadding3D, " +
        "padding can not be null, please input int tuple of length 3")
      case _ => Array(padding._1, padding._2, padding._3)
    }
    new ZeroPadding3D[T](paddingArray, KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
