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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class Cropping3D[T: ClassTag](
   val cropping: ((Int, Int), (Int, Int), (Int, Int)) = ((1, 1), (1, 1), (1, 1)),
   val format: String = "CHANNEL_FIRST",
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(format.toLowerCase() == "channel_first" || format.toLowerCase() == "channel_last",
  "Cropping3D only supports format channel_first or channel_last")

  private val dimOrdering = format.toLowerCase() match {
    case "channel_first" => com.intel.analytics.bigdl.nn.Cropping3D.CHANNEL_FIRST
    case "channel_last" => com.intel.analytics.bigdl.nn.Cropping3D.CHANNEL_LAST
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer =
      com.intel.analytics.bigdl.nn.Cropping3D(
        dim1Crop = Array(cropping._1._1, cropping._1._2),
        dim2Crop = Array(cropping._2._1, cropping._2._2),
        dim3Crop = Array(cropping._3._1, cropping._3._2),
        format = dimOrdering
      )
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Cropping3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    cropping: ((Int, Int), (Int, Int), (Int, Int)) = ((1, 1), (1, 1), (1, 1)),
    format: String = "CHANNEL_FIRST",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Cropping3D[T] = {
    new Cropping3D[T](cropping, format, inputShape)
  }
}
