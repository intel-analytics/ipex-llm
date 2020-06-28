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
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Zero-padding layer for 3D data (spatial or spatio-temporal).
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param padding Int array of length 3.
 *                How many zeros to add at the beginning and end of the 3 padding dimensions.
 *                Symmetric padding will be applied to each dimension. Default is (1, 1, 1).
 * @param dimOrdering Format of the input data. Either "CHANNEL_FIRST" (dimOrdering='th') or
 *                    "CHANNEL_LAST" (dimOrdering='tf'). Default is "CHANNEL_FIRST".
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class ZeroPadding3D[T: ClassTag](
   val padding: Array[Int] = Array(1, 1, 1),
   val dimOrdering: String = "CHANNEL_FIRST",
   val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering.toLowerCase() == "channel_first" ||
    dimOrdering.toLowerCase() == "channel_last",
    s"For ZeroPadding3D $dimOrdering is not supported")
  require(padding.length == 3, s"For ZeroPadding3D, subsample should be of length 3," +
    s" but got length ${padding.length}")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"ZeroPadding3D requires 5D input, but got input dim ${input.length}")
    dimOrdering.toLowerCase() match {
      case "channel_first" =>
        Shape(input(0), input(1), input(2) + 2 * padding(0),
          input(3) + 2 * padding(1), input(4) + 2 * padding(2))
      case "channel_last" =>
        Shape(input(0), input(1) + 2 * padding(0), input(2) + 2 * padding(1),
          input(3) + 2 * padding(2), input(4))
    }
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val dim = if (dimOrdering.toLowerCase() == "channel_first") 2 else 1
    val model = TSequential[T]()
    val paddinglayer1 = Padding(dim = dim, pad = -padding(0), nInputDim = input.length - 1)
    val paddinglayer2 = Padding(dim = dim, pad = padding(0), nInputDim = input.length - 1)
    val paddinglayer3 = Padding(dim = dim + 1, pad = -padding(1), nInputDim = input.length - 1)
    val paddinglayer4 = Padding(dim = dim + 1, pad = padding(1), nInputDim = input.length - 1)
    val paddinglayer5 = Padding(dim = dim + 2, pad = -padding(2), nInputDim = input.length - 1)
    val paddinglayer6 = Padding(dim = dim + 2, pad = padding(2), nInputDim = input.length - 1)
    model.add(paddinglayer1)
    model.add(paddinglayer2)
    model.add(paddinglayer3)
    model.add(paddinglayer4)
    model.add(paddinglayer5)
    model.add(paddinglayer6)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object ZeroPadding3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    padding: (Int, Int, Int) = (1, 1, 1),
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : ZeroPadding3D[T] = {
    new ZeroPadding3D[T](Array(padding._1, padding._2, padding._3),
      KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}
