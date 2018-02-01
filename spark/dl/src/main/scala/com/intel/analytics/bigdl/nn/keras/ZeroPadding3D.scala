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

class ZeroPadding3D[T: ClassTag](val padding: (Int, Int, Int) = (1, 1, 1),
                                 val format: String = "CHANNEL_FIRST",
                                 var inputShape: Shape = null
  )(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(format.toLowerCase() == "channel_first" || format.toLowerCase() == "channel_last",
    s"$format is not supported")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5, "ZeroPadding3D requires 5D input")
    format.toLowerCase() match {
      case "channel_first" =>
        Shape(input(0), input(1), input(2) + 2 * padding._1,
          input(3) + 2 * padding._2, input(4) + 2 * padding._3)
      case "channel_last" =>
        Shape(input(0), input(1) + 2 * padding._1, input(2) + 2 * padding._2,
          input(3) + 2 * padding._3, input(4))
    }
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val dim = if (format.toLowerCase == "channel_first") 2 else 1

    val model = TSequential[T]()
    val paddinglayer1 = Padding(
      dim = dim,
      pad = -padding._1,
      nInputDim = input.length - 1,
      value = 0.0,
      nIndex = 1)

    val paddinglayer2 = Padding(
      dim = dim,
      pad = padding._1,
      nInputDim = input.length - 1,
      value = 0.0,
      nIndex = 1)

    val paddinglayer3 = Padding(
      dim = dim + 1,
      pad = -padding._2,
      nInputDim = input.length - 1,
      value = 0.0,
      nIndex = 1)

    val paddinglayer4 = Padding(
      dim = dim + 1,
      pad = padding._2,
      nInputDim = input.length - 1,
      value = 0.0,
      nIndex = 1)

    val paddinglayer5 = Padding(
      dim = dim + 2,
      pad = -padding._3,
      nInputDim = input.length - 1,
      value = 0.0,
      nIndex = 1)

    val paddinglayer6 = Padding(
      dim = dim + 2,
      pad = padding._3,
      nInputDim = input.length - 1,
      value = 0.0,
      nIndex = 1)

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
    format: String = "CHANNEL_FIRST",
    inputShape: Shape = null
    )(implicit ev: TensorNumeric[T]) : ZeroPadding3D[T] = {
    new ZeroPadding3D[T](
      padding,
      format,
      inputShape)
  }
}
