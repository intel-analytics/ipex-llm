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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

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
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Cropping2D[T: ClassTag](
   val heightCrop: Array[Int] = Array(0, 0),
   val widthCrop: Array[Int] = Array(0, 0),
   val dimOrdering: DataFormat = DataFormat.NCHW,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(heightCrop.length == 2,
    s"Cropping3D: height cropping values should be of length 2, but got ${heightCrop.length}")
  require(widthCrop.length == 2,
    s"Cropping3D: width cropping values should be of length 2, but got ${widthCrop.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.Cropping2D(
      heightCrop = heightCrop,
      widthCrop = widthCrop,
      format = dimOrdering)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Cropping2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: ((Int, Int), (Int, Int)) = ((0, 0), (0, 0)),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    val heightCrop = Array(cropping._1._1, cropping._1._2)
    val widthCrop = Array(cropping._2._1, cropping._2._2)
    new Cropping2D[T](heightCrop, widthCrop,
      KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
