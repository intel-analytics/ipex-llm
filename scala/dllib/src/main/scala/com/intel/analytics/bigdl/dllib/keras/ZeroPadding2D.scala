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

import com.intel.analytics.bigdl.nn.Padding
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

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
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ZeroPadding2D[T: ClassTag](
   val padding: Array[Int] = Array(1, 1, 1, 1),
   val dimOrdering: DataFormat = DataFormat.NCHW,
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(padding.length == 4,
    s"For ZeroPadding2D, padding values should be of length 4 " +
      s"(top_pad, bottom_pad, left_pad, right_pad), but got length ${padding.length}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"ZeroPadding2D requires 4D input, but got input dim ${input.length}")
    dimOrdering match {
      case DataFormat.NCHW =>
        Shape(input(0), input(1),
          input(2) + padding(0) + padding(1), input(3) + padding(2) + padding(3))
      case DataFormat.NHWC =>
        Shape(input(0), input(1) + padding(0) + padding(1),
          input(2) + padding(2) + padding(3), input(3))
    }
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val nInputDim = input.length -1
    val (dim1, dim2) = dimOrdering match {
      case DataFormat.NCHW => (2, 3)
      case DataFormat.NHWC => (1, 2)
    }
    val model = TSequential[T]()
    val pad1 = Padding(dim1, -padding(0), nInputDim)
    val pad2 = Padding(dim1, padding(1), nInputDim)
    val pad3 = Padding(dim2, -padding(2), nInputDim)
    val pad4 = Padding(dim2, padding(3), nInputDim)
    model.add(pad1)
    model.add(pad2)
    model.add(pad3)
    model.add(pad4)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ZeroPadding2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    padding: (Int, Int) = (1, 1),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ZeroPadding2D[T] = {
    new ZeroPadding2D[T](Array(padding._1, padding._1, padding._2, padding._2),
      KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}
