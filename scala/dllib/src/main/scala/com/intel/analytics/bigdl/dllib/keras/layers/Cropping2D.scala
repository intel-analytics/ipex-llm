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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.dllib.nn.keras.{Cropping2D => BigDLCropping2D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Cropping layer for 2D input (e.g. picture).
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param heightCrop Int array of length 2. Height of the 2 cropping dimension. Default is (0, 0).
 * @param widthCrop Int array of length 2. Width of the 2 cropping dimension. Default is (0, 0).
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Cropping2D[T: ClassTag](
    override val heightCrop: Array[Int] = Array(0, 0),
    override val widthCrop: Array[Int] = Array(0, 0),
    override val dimOrdering: DataFormat = DataFormat.NCHW,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLCropping2D[T](
    heightCrop, widthCrop, dimOrdering, inputShape) with Net {
}

object Cropping2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: ((Int, Int), (Int, Int)) = ((0, 0), (0, 0)),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    require(cropping != null, "For Cropping2D, " +
      "cropping values should be int tuple of tuple of length 2")
    require(cropping._1 != null, "For Cropping2D, heightCrop should be int tuple of length 2")
    require(cropping._2 != null, "For Cropping2D, widthCrop should be int tuple of length 2")
    val heightCrop = Array(cropping._1._1, cropping._1._2)
    val widthCrop = Array(cropping._2._1, cropping._2._2)
    new Cropping2D[T](heightCrop, widthCrop,
      KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
