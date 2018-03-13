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

/**
 * UpSampling layer for 3D inputs.
 * Repeats the 1st, 2nd and 3rd dimensions of the data by size(0), size(1) and size(2) respectively.
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param size Int array of length 3. UpSampling factors for dim1, dim2 and dim3.
 *             Default is (2, 2, 2).
 * @param dimOrdering Format of the input data. Please use "CHANNEL_FIRST" (dimOrdering='th').
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class UpSampling3D[T: ClassTag](
   val size: Array[Int] = Array(2, 2, 2),
   val dimOrdering: String = "CHANNEL_FIRST",
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering.toLowerCase() == "channel_first",
    s"UpSampling3D currently only supports format CHANNEL_FIRST, but got format $dimOrdering")
  require(size.length == 3,
    s"UpSampling3D: upsampling sizes should be of length 3, but got ${size.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.UpSampling3D(size)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object UpSampling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: (Int, Int, Int) = (2, 2, 2),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): UpSampling3D[T] = {
    new UpSampling3D[T](Array(size._1, size._2, size._3),
      KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
