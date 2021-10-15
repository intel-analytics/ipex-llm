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

import com.intel.analytics.bigdl.dllib.nn.keras.{UpSampling2D => BigDLUpSampling2D}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * UpSampling layer for 2D inputs.
 * Repeats the rows and columns of the data by size(0) and size(1) respectively.
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param size Int array of length 2. UpSampling factors for rows and columns.
 *             Default is (2, 2).
 * @param dimOrdering Format of the input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class UpSampling2D[T: ClassTag](
   override val size: Array[Int] = Array(2, 2),
   override val dimOrdering: DataFormat = DataFormat.NCHW,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLUpSampling2D[T](size, dimOrdering, inputShape) with Net {}

object UpSampling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: (Int, Int) = (2, 2),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): UpSampling2D[T] = {
    val sizeArray = size match {
      case null => throw new IllegalArgumentException("For UpSampling2D, " +
        "size can not be null, please input int tuple of length 2")
      case _ => Array(size._1, size._2)
    }
    new UpSampling2D[T](sizeArray, KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
