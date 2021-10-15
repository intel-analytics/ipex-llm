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
import com.intel.analytics.bigdl.dllib.nn.keras.{KerasLayer, ZeroPadding2D => BigDLZeroPadding2D}
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.dllib.nn.Padding
import com.intel.analytics.bigdl.dllib.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Net

import scala.reflect.ClassTag

/**
 * Zero-padding layer for 2D input (e.g. picture).
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param padding Int array of length 4.
 *                How many zeros to add at the beginning and at the end of the 2 padding dimensions
 *                (rows and cols), in the order '(top_pad, bottom_pad, left_pad, right_pad)'.
 *                Default is (1, 1, 1, 1).
 * @param dimOrdering Format of the input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ZeroPadding2D[T: ClassTag](
   override val padding: Array[Int] = Array(1, 1, 1, 1),
   override val dimOrdering: DataFormat = DataFormat.NCHW,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLZeroPadding2D[T](padding, dimOrdering, inputShape) with Net {}

object ZeroPadding2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    padding: (Int, Int) = (1, 1),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ZeroPadding2D[T] = {
    val paddingArray = padding match {
      case null => throw new IllegalArgumentException("For ZeroPadding2D, " +
        "padding can not be null, please input int tuple of length 2")
      case _ => Array(padding._1, padding._1, padding._2, padding._2)
    }
    new ZeroPadding2D[T](
      paddingArray, KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
