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

import com.intel.analytics.bigdl.dllib.nn.internal.{Cropping3D => BigDLCropping3D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Cropping layer for 3D data (e.g. spatial or spatio-temporal).
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dim1Crop Int array of length 2. Kernel dim1 of the three cropping dimensions.
 *                 Default is (1, 1).
 * @param dim2Crop Int array of length 2. Kernel dim2 of the three cropping dimensions.
 *                 Default is (1, 1).
 * @param dim3Crop Int array of length 2. Kernel dim3 of the three cropping dimensions.
 *                 Default is (1, 1).
 * @param dimOrdering Format of input data. Either 'CHANNEL_FIRST' (dimOrdering='th') or
 *                    'CHANNEL_LAST' (dimOrdering='tf'). Default is 'CHANNEL_FIRST'.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Cropping3D[T: ClassTag](
    override val dim1Crop: Array[Int] = Array(1, 1),
    override val dim2Crop: Array[Int] = Array(1, 1),
    override val dim3Crop: Array[Int] = Array(1, 1),
    override val dimOrdering: String = "CHANNEL_FIRST",
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLCropping3D[T](
    dim1Crop, dim2Crop, dim3Crop, dimOrdering, inputShape) with Net {
}

object Cropping3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: ((Int, Int), (Int, Int), (Int, Int)) = ((1, 1), (1, 1), (1, 1)),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping3D[T] = {
    require(cropping != null, "For Cropping3D, " +
      "cropping values should be int tuple of tuple of length 2")
    require(cropping._1 != null, "For Cropping3D, dim1Crop should be int tuple of length 2")
    require(cropping._2 != null, "For Cropping3D, dim2Crop should be int tuple of length 2")
    require(cropping._3 != null, "For Cropping3D, dim3Crop should be int tuple of length 2")
    val dim1Crop = Array(cropping._1._1, cropping._1._2)
    val dim2Crop = Array(cropping._2._1, cropping._2._2)
    val dim3Crop = Array(cropping._3._1, cropping._3._2)
    new Cropping3D[T](dim1Crop, dim2Crop, dim3Crop,
      KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
