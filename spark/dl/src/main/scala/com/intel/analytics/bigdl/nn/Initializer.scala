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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

/**
 * Initialization method to initialize bias and weight.
 * The init method will be called in Module.reset()
 */

trait Initializer {

  type Shape = Array[Int]

  def init[T](weight: Tensor[T], bias: Option[Tensor[T]], fmt: String = "output_first")
          (implicit ev: TensorNumeric[T]): Unit

  protected def getFans(shape: Shape, dataFormat: String = "output_first"): (Int, Int) = {
    val dims = shape.length
    val (first, second) = dims match {
      case 2 => (shape(0), shape(1))
      case 4 =>
        val receptiveFieldSize = shape(2) * shape(3)
        (shape(0) * receptiveFieldSize, shape(1) * receptiveFieldSize)
      case 5 =>
        val receptiveFieldSize = shape(0) * shape(2) * shape(3)
        (shape(0) * receptiveFieldSize, shape(1) * receptiveFieldSize)
      case _ =>
        val sqrtElem = Math.sqrt(shape.product).toInt
        (sqrtElem, sqrtElem)
    }
    val (fanIn, fanOut) = if (dataFormat == "input_first") {
      (first, second)
    } else if (dataFormat == "output_first") {
      (second, first)
    } else {
      throw new IllegalArgumentException(s"Invalid inputFormat: $dataFormat")
    }
    (Math.max(1, fanIn), Math.max(1, fanOut))
  }
}

case object RandomUniform extends Initializer {

  def init[T](weight: Tensor[T], bias: Option[Tensor[T]], fmt: String = "output_first")
          (implicit ev: TensorNumeric[T]): Unit = {
    val shape = weight.size()
    val (fanIn, _) = getFans(shape, fmt)
    val stdv = 1.0 / math.sqrt(fanIn)
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
    bias.foreach(_.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv))))
  }

}

/**
 * In short, it helps signals reach deep into the network.
 *
 * During the training process of deep nn:
 *        1. If the weights in a network start are too small,
 *           then the signal shrinks as it passes through
 *           each layer until it’s too tiny to be useful.
 *
 *        2. If the weights in a network start too large,
 *           then the signal grows as it passes through each
 *           layer until it’s too massive to be useful.
 *
 * Xavier initialization makes sure the weights are ‘just right’,
 * keeping the signal in a reasonable range of values through many layers.
 *
 * More details on the paper
 *  [Understanding the difficulty of training deep feedforward neural networks]
 *  (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
 */
case object Xavier extends Initializer {
  def init[T](weight: Tensor[T], bias: Option[Tensor[T]], fmt: String = "output_first")
          (implicit ev: TensorNumeric[T]): Unit = {
    val shape = weight.size()
    val (fanIn, fanOut) = getFans(shape)
    val stdv = math.sqrt(6.0 / (fanIn + fanOut))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
    bias.foreach(_.fill(ev.fromType(0)))
  }

}
case object BilinearFiller extends Initializer {
  def init[T](weight: Tensor[T], bias: Option[Tensor[T]], fmt: String = "output_first")
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = weight.size()
    require(shape.length == 5, s"SpatialFullConvolution: weight must be 5 dim, " +
      s"but got ${shape.length}")
    val kH = shape(3)
    val kW = shape(4)
    require(kH == kW, s"SpatialFullConvolution: Kernel $kH * $kW must be square")
    val f = Math.ceil(kW / 2.0).toInt
    val c = (2 * f - 1 - f % 2) / (2.0f * f)
    val weightArray = weight.storage().array()
    val weightOffset = weight.storageOffset() - 1
    var i = 0
    while(i < weight.nElement()) {
      val x : Float = i % kW
      val y : Float = (i / kW) % kH
      weightArray(i + weightOffset) = ev.fromType[Float](
        (1f - math.abs(x / f - c)) * (1f - math.abs(y / f - c)))
      i += 1
    }
  }
}
