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
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class UpSampling2D[T: ClassTag](
   val size: Array[Int] = Array(2, 2),
   val dimOrdering: DataFormat = DataFormat.NCHW,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(size.length == 2,
    s"UpSampling2D: upsampling sizes should be of length 2, but got ${size.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.UpSampling2D(
      size = Array(size(0), size(1)),
      format = dimOrdering)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object UpSampling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: (Int, Int) = (2, 2),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): UpSampling2D[T] = {
    new UpSampling2D[T](Array(size._1, size._2),
      KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
