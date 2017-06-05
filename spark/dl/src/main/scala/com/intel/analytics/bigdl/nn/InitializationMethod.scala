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
 * DataFormat describe the dimension of weights
 */
sealed trait DataFormat

/**
 * OutputFirst indicating that output is in the lower dimension
 * of the weight. Take 2 dimensional weight for example, the first
 * dimension is the output, and the second dimension is the input.
 */
case object OutputFirst extends DataFormat

/**
 * InputFirst indicating that input is in the lower dimension
 * of the weight. Take 2 dimensional weight for example, the first
 * dimension is the input, and the second dimension is the output.
 */
case object InputFirst extends DataFormat



/**
 * Initialization method to initialize bias and weight.
 * The init method will be called in Module.reset()
 */

trait InitializationMethod {

  type Shape = Array[Int]

  /**
   * Initialize the given weight and bias.
   *
   * @param weight    the weight to initialize
   * @param dataFormat       the data format of weight indicating the dimension order of
   *                  the weight. "output_first" means output is in the lower dimension
   *                  "input_first" means input is in the lower dimension.
   */
  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit

  /**
   * Get the fanIn and fanOut from shape.
   *
   * This method takes the following assumptions on shape.
   *  1. If shape has 2 dimensions, these two dimensions are the input and output.
   *  2. If shape has 4 dimensions (convolutional weight), the first two dimensions
   *     are input and output. The 3th and 4th dimensions are the kernel width and kernel height.
   *  3. If shape has 5 dimensions (convolutional weight), the first dimension is
   *  group size, the 2nd and 3th dimensions are input and output, and the last two dimensions
   *  are kernel width and kernel height.
   *
   * @param shape the shape of which to get the fans
   * @param dataFormat the dimension order of the shape
   * @return the (fanIn, fanOut) tuple
   */
  protected def getFans(shape: Shape, dataFormat: DataFormat = OutputFirst): (Int, Int) = {
    val dims = shape.length
    val (first, second) = dims match {
      case 2 => (shape(0), shape(1))
      case 4 =>
        val receptiveFieldSize = shape(2) * shape(3)
        (shape(0) * receptiveFieldSize, shape(1) * receptiveFieldSize)
      case 5 =>
        val receptiveFieldSize = shape(0) * shape(3) * shape(4)
        (shape(1) * receptiveFieldSize, shape(2) * receptiveFieldSize)
      case _ =>
        val sqrtElem = Math.sqrt(shape.product).toInt
        (sqrtElem, sqrtElem)
    }
    val (fanIn, fanOut) = dataFormat match {
      case InputFirst => (first, second)
      case OutputFirst => (second, first)
      case _ =>
        throw new IllegalArgumentException(s"Invalid inputFormat: $dataFormat")
    }
    (Math.max(1, fanIn), Math.max(1, fanOut))
  }

  protected def initWithUniform[T](tensor: Tensor[T], stdv: Double)
                                  (implicit ev: TensorNumeric[T]): Unit = {
    tensor.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }
}


/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * It draws samples form a uniform distribution within [-limit, limit]
 * where "limit" is "1/sqrt(fan_in)"
 *
 */
case object RandomUniform extends InitializationMethod {

  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = weight.size()
    val (fanIn, _) = getFans(shape, dataFormat)
    val stdv = 1.0 / math.sqrt(fanIn)
    initWithUniform(weight, stdv)
  }

}

/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * It draws samples form a uniform distribution within [-limit, limit]
 * where "limit" is specified by stdv
 *
 */
case class RandomUniform(stdv: Double) extends InitializationMethod {

  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    initWithUniform(weight, stdv)
  }

}

/**
 * Initializer that generates tensors with zeros.
 */
case object Zeros extends InitializationMethod {

  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    weight.zero()
  }

}

/**
 * Initializer that generates tensors with zeros.
 */
case object Ones extends InitializationMethod {

  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    weight.fill(ev.one)
  }
}

/**
 * Initializer that generates tensors with certain constant double.
 */
case class Const(value: Double) extends InitializationMethod {

  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    weight.fill(ev.fromType(value))
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
case object Xavier extends InitializationMethod {
  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = weight.size()
    val (fanIn, fanOut) = getFans(shape)
    val stdv = math.sqrt(6.0 / (fanIn + fanOut))
    initWithUniform(weight, stdv)
  }

}

/**
 * Initialize the weight with coefficients for bilinear interpolation.
 *
 * A common use case is with the DeconvolutionLayer acting as upsampling.
 *
 */
case object BilinearFiller extends InitializationMethod {
  def init[T](weight: Tensor[T], dataFormat: DataFormat = OutputFirst)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = weight.size()
    require(shape.length == 5, s"weight must be 5 dim, " +
      s"but got ${shape.length}")
    val kH = shape(3)
    val kW = shape(4)
    require(kH == kW, s"Kernel $kH * $kW must be square")
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
