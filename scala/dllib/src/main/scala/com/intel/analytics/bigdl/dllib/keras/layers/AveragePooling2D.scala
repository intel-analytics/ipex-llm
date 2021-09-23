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

import com.intel.analytics.bigdl.dllib.nn.SpatialAveragePooling
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.nn.keras.{KerasLayer, Pooling2D}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies average pooling operation for spatial data.
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
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class AveragePooling2D[T: ClassTag](
    override val poolSize: Array[Int] = Array(2, 2),
    override val strides: Array[Int] = null,
    override val borderMode: String = "valid",
    val dimOrdering: DataFormat = DataFormat.NCHW,
    override val inputShape: Shape = null,
    val pads: Array[Int] = null,
    val countIncludePad: Boolean = false)(implicit ev: TensorNumeric[T])
  extends Pooling2D[T](poolSize, strides, borderMode, inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    val paddings = KerasUtils.getPadsFromBorderMode(borderMode, pads)
    val layer = SpatialAveragePooling(
      kW = poolSize(1),
      kH = poolSize(0),
      dW = strideValues(1),
      dH = strideValues(0),
      padW = paddings._2,
      padH = paddings._1,
      countIncludePad = false,
      format = dimOrdering)
    layer.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object AveragePooling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: (Int, Int) = (2, 2),
    strides: (Int, Int) = null,
    borderMode: String = "valid",
    dimOrdering: String = "th",
    inputShape: Shape = null,
    pads: Array[Int] = null,
    countIncludePad: Boolean = false)(implicit ev: TensorNumeric[T]): AveragePooling2D[T] = {
    val poolSizeArray = poolSize match {
      case null => throw new IllegalArgumentException("For AveragePooling2D, " +
        "poolSize can not be null, please input int tuple of length 2")
      case _ => Array(poolSize._1, poolSize._2)
    }
    val strideArray = strides match {
      case null => null
      case _ => Array(strides._1, strides._2)
    }
    new AveragePooling2D[T](
      poolSizeArray, strideArray,
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), inputShape, pads, countIncludePad)
  }
}
