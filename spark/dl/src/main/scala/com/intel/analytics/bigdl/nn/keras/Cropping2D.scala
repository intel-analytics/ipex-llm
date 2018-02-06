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

class Cropping2D[T: ClassTag](
   val cropping: Array[Array[Int]] = Array(Array(0, 0), Array(0, 0)),
   val format: DataFormat = DataFormat.NCHW,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(cropping.length == 2,
    s"Cropping2D requires two cropping dimensions, but got ${cropping.length}")
  require(cropping(0).length == 2,
    s"Cropping values in height dimension should be of length 2, but got ${cropping(0).length}")
  require(cropping(1).length == 2,
    s"Cropping values in width dimension should be of length 2, but got ${cropping(1).length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer =
      com.intel.analytics.bigdl.nn.Cropping2D(
        heightCrop = cropping(0),
        widthCrop = cropping(1),
        format = format)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Cropping2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: ((Int, Int), (Int, Int)) = ((0, 0), (0, 0)),
    format: DataFormat = DataFormat.NCHW,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping2D[T] = {
    val heightCrop = Array(cropping._1._1, cropping._1._2)
    val widthCrop = Array(cropping._2._1, cropping._2._2)
    new Cropping2D[T](Array(heightCrop, widthCrop), format, inputShape)
  }
}
