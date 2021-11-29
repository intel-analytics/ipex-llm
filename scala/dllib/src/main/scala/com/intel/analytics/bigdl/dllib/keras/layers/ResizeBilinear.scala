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

import com.intel.analytics.bigdl.dllib.{nn => bnn}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Resize the input image with bilinear interpolation. The input image must be a float tensor with
 * NHWC or NCHW layout.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param outputHeight output height
 * @param outputWidth output width
 * @param alignCorners align corner or not
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class ResizeBilinear[T: ClassTag](
  val outputHeight: Int,
  val outputWidth: Int,
  val alignCorners: Boolean,
  val dimOrdering: DataFormat = DataFormat.NCHW,
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends LayerWrapperByForward[T](KerasUtils.addBatch(inputShape))  {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    bnn.ResizeBilinear[T](outputHeight, outputWidth, alignCorners, dimOrdering)
  }
}

object ResizeBilinear {

  def apply[T: ClassTag](
      outputHeight: Int,
      outputWidth: Int,
      alignCorners: Boolean = false,
      dimOrdering: String = "th",
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ResizeBilinear[T] = {
    new ResizeBilinear[T](outputHeight, outputWidth, alignCorners,
      KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
