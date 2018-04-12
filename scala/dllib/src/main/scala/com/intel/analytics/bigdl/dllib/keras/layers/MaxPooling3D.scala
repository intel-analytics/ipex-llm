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

import com.intel.analytics.bigdl.nn.keras.{MaxPooling3D => BigDLMaxPooling3D}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies max pooling operation for 3D data (spatial or spatio-temporal).
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * Border mode currently supported for this layer is 'valid'.
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param poolSize Int array of length 3. Factors by which to downscale (dim1, dim2, dim3).
 *                 Default is (2, 2, 2), which will halve the image in each dimension.
 * @param strides Int array of length 3. Stride values. Default is null, and in this case it will
 *                be equal to poolSize.
 * @param dimOrdering Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class MaxPooling3D[T: ClassTag](
    poolSize: Array[Int] = Array(2, 2, 2),
    strides: Array[Int] = null,
    dimOrdering: String = "CHANNEL_FIRST",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLMaxPooling3D[T](
    poolSize, strides, dimOrdering, inputShape) {
}

object MaxPooling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: (Int, Int, Int) = (2, 2, 2),
    strides: (Int, Int, Int) = null,
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): MaxPooling3D[T] = {
    val strideArray = strides match {
      case null => null
      case _ => Array(strides._1, strides._2, strides._3)
    }
    val poolSizeArray = poolSize match {
      case null => throw new IllegalArgumentException("" +
        "poolSize can not be null, please input int tuple of length 3.")
      case _ => Array(poolSize._1, poolSize._2, poolSize._3)
    }
    new MaxPooling3D[T](poolSizeArray, strideArray,
      KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
