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

import com.intel.analytics.bigdl.dllib.nn.internal.{UpSampling3D => BigDLUpSampling3D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * UpSampling layer for 3D inputs.
 * Repeats the 1st, 2nd and 3rd dimensions of the data by size(0), size(1) and size(2) respectively.
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param size Int array of length 3. UpSampling factors for dim1, dim2 and dim3.
 *             Default is (2, 2, 2).
 * @param dimOrdering Format of the input data. Please use "CHANNEL_FIRST" (dimOrdering='th').
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class UpSampling3D[T: ClassTag](
   override val size: Array[Int] = Array(2, 2, 2),
   override val dimOrdering: String = "CHANNEL_FIRST",
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLUpSampling3D[T](size, dimOrdering, inputShape) with Net {}

object UpSampling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: (Int, Int, Int) = (2, 2, 2),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): UpSampling3D[T] = {
    val sizeArray = size match {
      case null => throw new IllegalArgumentException("For UpSampling3D, " +
        "size can not be null, please input int tuple of length 3")
      case _ => Array(size._1, size._2, size._3)
    }
    new UpSampling3D[T](sizeArray, KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
