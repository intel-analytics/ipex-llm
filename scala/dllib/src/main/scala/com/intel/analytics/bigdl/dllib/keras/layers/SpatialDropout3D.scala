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

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.nn.keras.{SpatialDropout3D => BigDLSpatialDropout3D}
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Spatial 3D version of Dropout.
 * This version performs the same function as Dropout, however it drops
 * entire 3D feature maps instead of individual elements. If adjacent voxels
 * within feature maps are strongly correlated (as is normally the case in
 * early convolution layers) then regular dropout will not regularize the
 * activations and will otherwise just result in an effective learning rate
 * decrease. In this case, SpatialDropout3D will help promote independence
 * between feature maps and should be used instead.
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param p Fraction of the input units to drop. Double between 0 and 1.
 * @param dimOrdering Format of input data. Either 'CHANNEL_FIRST' (dimOrdering='th') or
 *                    'CHANNEL_LAST' (dimOrdering='tf'). Default is 'CHANNEL_FIRST'.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SpatialDropout3D[T: ClassTag](
    override val p: Double = 0.5,
    override val dimOrdering: String = "CHANNEL_FIRST",
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLSpatialDropout3D[T](
    p, dimOrdering, inputShape) with Net {
}

object SpatialDropout3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double = 0.5,
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SpatialDropout3D[T] = {
    new SpatialDropout3D[T](p, KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
