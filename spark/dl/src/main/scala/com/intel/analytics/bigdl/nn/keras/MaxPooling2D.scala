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

import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Applies max pooling operation for spatial data.
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param poolSize Int array of length 2 corresponding to the downscale vertically and
 *                 horizontally. Default is (2, 2), which will halve the image in each dimension.
 * @param strides Int array of length 2. Stride values. Default is null, and in this case it will
 *                be equal to poolSize.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class MaxPooling2D[T: ClassTag] (
   poolSize: Array[Int] = Array(2, 2),
   strides: Array[Int] = null,
   borderMode: String = "valid",
   dimOrdering: DataFormat = DataFormat.NCHW,
   inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Pooling2D[T](poolSize, strides, borderMode, dimOrdering, inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val pads = KerasUtils.getPadsFromBorderMode(borderMode)
    val layer = SpatialMaxPooling(
      kW = poolSize(1),
      kH = poolSize(0),
      dW = strideValues(1),
      dH = strideValues(0),
      padW = pads._2,
      padH = pads._1,
      format = dimOrdering)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object MaxPooling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: (Int, Int) = (2, 2),
    strides: (Int, Int) = null,
    borderMode: String = "valid",
    dimOrdering: String = "th",
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): MaxPooling2D[T] = {
    val strideValues = if (strides != null) Array(strides._1, strides._2) else null
    new MaxPooling2D[T](Array(poolSize._1, poolSize._2), strideValues,
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
