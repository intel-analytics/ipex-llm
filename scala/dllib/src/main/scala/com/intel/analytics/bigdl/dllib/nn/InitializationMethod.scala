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

import com.intel.analytics.bigdl.nn.VariableFormat.Default
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator

/**
 * VariableFormat describe the meaning of each dimension of the variable
 * (the trainable parameters of a model like weight and bias) and can be used to
 * return the fan in and fan out size of the variable when provided the variable shape.
 */
trait VariableFormat {
  def getFanIn(shape: Array[Int]): Int = {
    throw new Exception("FanIn is not defined in this format")
  }
  def getFanOut(shape: Array[Int]): Int = {
    throw new Exception("FanOut is not defined in this format")
  }
}

object VariableFormat {

  /**
   * The default VariableFormat used when we do not care about
   * the specified format of this variable.
   */
  case object Default extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      shape.product
    }

    override def getFanOut(shape: Array[Int]): Int = {
      shape.product
    }

  }

  case object ONE_D extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      shape(0)
    }

    override def getFanOut(shape: Array[Int]): Int = {
      1
    }
  }

  case object IN_OUT extends VariableFormat {

    override def getFanIn(shape: Array[Int]): Int = {
      shape(0)
    }

    override def getFanOut(shape: Array[Int]): Int = {
      shape(1)
    }
  }
  case object OUT_IN extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      shape(1)
    }

    override def getFanOut(shape: Array[Int]): Int = {
      shape(0)
    }
  }
  case object IN_OUT_KW_KH extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(2) * shape(3)
      shape(0) * receptiveFieldSize
    }

    override def getFanOut(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(2) * shape(3)
      shape(1) * receptiveFieldSize
    }
  }

  case object OUT_IN_KW_KH extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(2) * shape(3)
      shape(1) * receptiveFieldSize
    }

    override def getFanOut(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(2) * shape(3)
      shape(0) * receptiveFieldSize
    }
  }

  case object GP_OUT_IN_KW_KH extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(0) * shape(3) * shape(4)
      shape(2) * receptiveFieldSize
    }

    override def getFanOut(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(0) * shape(3) * shape(4)
      shape(1) * receptiveFieldSize
    }
  }

  case object GP_IN_OUT_KW_KH extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(0) * shape(3) * shape(4)
      shape(1) * receptiveFieldSize
    }

    override def getFanOut(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(0) * shape(3) * shape(4)
      shape(2) * receptiveFieldSize
    }
  }

  case object OUT_IN_KT_KH_KW extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(2) * shape(3) * shape(4)
      shape(1) * receptiveFieldSize
    }

    override def getFanOut(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(2) * shape(3) * shape(4)
      shape(0) * receptiveFieldSize
    }
  }

  case object GP_KH_KW_IN_OUT extends VariableFormat {
    override def getFanIn(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(0) * shape(1) * shape(2)
      shape(2) * receptiveFieldSize
    }

    override def getFanOut(shape: Array[Int]): Int = {
      val receptiveFieldSize = shape(0) * shape(1) * shape(2)
      shape(3) * receptiveFieldSize
    }
  }
}

/**
 * Initialization method to initialize bias and weight.
 * The init method will be called in Module.reset()
 */

trait InitializationMethod {

  type Shape = Array[Int]

  /**
   * Initialize the given weight and bias.
   *
   * @param variable    the weight to initialize
   * @param dataFormat       the data format of weight indicating the dimension order of
   *                  the weight. "output_first" means output is in the lower dimension
   *                  "input_first" means input is in the lower dimension.
   */
  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit
}


/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * It draws samples from a uniform distribution within [-limit, limit]
 * where "limit" is "1/sqrt(fan_in)"
 *
 */
case object RandomUniform extends InitializationMethod {

  override def init[T](variable: Tensor[T], dataFormat: VariableFormat)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = variable.size()
    val fanIn = dataFormat.getFanIn(shape)
    val stdv = 1.0 / math.sqrt(fanIn)
    variable.rand(-stdv, stdv)
  }

}

/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * It draws samples from a uniform distribution within [lower, upper]
 *
 */
case class RandomUniform(lower: Double, upper: Double) extends InitializationMethod {

  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit = {
    variable.rand(lower, upper)
  }

}

/**
 * Initializer that generates tensors with a normal distribution.
 *
 */
case class RandomNormal(mean: Double, stdv: Double) extends InitializationMethod {

  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit = {
    variable.randn(mean, stdv)
  }

}

/**
 * Initializer that generates tensors with zeros.
 */
case object Zeros extends InitializationMethod {

  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit = {
    variable.zero()
  }

}

/**
 * Initializer that generates tensors with zeros.
 */
case object Ones extends InitializationMethod {

  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit = {
    variable.fill(ev.one)
  }
}

/**
 * Initializer that generates tensors with certain constant double.
 */
case class ConstInitMethod(value: Double) extends InitializationMethod {

  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit = {
    variable.fill(ev.fromType(value))
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
  private var varianceNormAverage: Boolean = true

  def setVarianceNormAverage(v: Boolean): this.type = {
    varianceNormAverage = v
    this
  }

  def init[T](variable: Tensor[T], dataFormat: VariableFormat)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = variable.size()
    val fanIn = dataFormat.getFanIn(shape)
    val fanOut = dataFormat.getFanOut(shape)
    val stdv = if (!varianceNormAverage) {
      math.sqrt(3.0 / fanIn)
    } else {
      math.sqrt(6.0 / (fanIn + fanOut))
    }
    variable.rand(-stdv, stdv)
  }

}

/**
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fanIn, fanOut, or their average, depending on
 * the varianceNormAverage parameter.
 *
 * @param varianceNormAverage VarianceNorm use average of (fanIn + fanOut) or just fanOut
 */
case class MsraFiller(varianceNormAverage: Boolean = true) extends InitializationMethod {
  def init[T](variable: Tensor[T], dataFormat: VariableFormat)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = variable.size()
    val fanIn = dataFormat.getFanIn(shape)
    val fanOut = dataFormat.getFanOut(shape)
    val n = if (varianceNormAverage) {
      (fanIn + fanOut) / 2
    } else {
      fanOut
    }
    val std = math.sqrt(2.0 / n)
    variable.apply1(_ => ev.fromType(RandomGenerator.RNG.normal(0, std)))
  }
}

/**
 * Initialize the weight with coefficients for bilinear interpolation.
 *
 * A common use case is with the DeconvolutionLayer acting as upsampling.
 * The variable tensor passed in the init function should have 5 dimensions
 * of format [nGroup, nInput, nOutput, kH, kW], and kH should be equal to kW
 *
 */
case object BilinearFiller extends InitializationMethod {
  def init[T](variable: Tensor[T], dataFormat: VariableFormat = Default)
             (implicit ev: TensorNumeric[T]): Unit = {
    val shape = variable.size()
    require(shape.length == 5, s"weight must be 5 dim, " +
      s"but got ${shape.length}")
    val kH = shape(3)
    val kW = shape(4)
    require(kH == kW, s"Kernel $kH * $kW must be square")
    val f = Math.ceil(kW / 2.0).toInt
    val c = (2 * f - 1 - f % 2) / (2.0f * f)
    val weightArray = variable.storage().array()
    val weightOffset = variable.storageOffset() - 1
    var i = 0
    while(i < variable.nElement()) {
      val x : Float = i % kW
      val y : Float = (i / kW) % kH
      weightArray(i + weightOffset) = ev.fromType[Float](
        (1f - math.abs(x / f - c)) * (1f - math.abs(y / f - c)))
      i += 1
    }
  }
}
