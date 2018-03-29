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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Apply multiplicative 1-centered Gaussian noise.
 * The multiplicative noise will have standard deviation `sqrt(rate / (1 - rate)).
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Output shape is the same as input.
 *
 *
 * @param rate double, drop probability (as with `Dropout`).
 *
 */

@SerialVersionUID(- 1575781981601306833L)
class GaussianDropout[T: ClassTag](
   val rate: Double
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(rate < 1 && rate >= 0, s"rate should be in range [0,1)")
  val stddev: Double = Math.sqrt(rate / (1.0-rate))

  override def updateOutput(input: Tensor[T]): Tensor[T] = {

    this.output.resizeAs(input).copy(input)

    if(train) {
      // generate a new random noise tensor in each forward and backward
      // following the behavior of tensorflow
      val noise = Tensor[T]()
      noise.resizeAs(input)
      noise.randn(1.0, stddev)
      this.output.cmul(noise)
    } else {
      this.output
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    this.gradInput.resizeAs(gradOutput).copy(gradOutput)

    if (train) {
      val noise = Tensor[T]()
      noise.resizeAs(gradOutput)
      noise.randn(1.0, stddev)
      this.gradInput.cmul(noise)
    } else {
      throw new IllegalArgumentException("backprop only defined while training")
    }
    this.gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($rate)"
  }



}

object GaussianDropout {
  def apply[@specialized(Float, Double) T: ClassTag](
    rate: Double
    )(implicit ev: TensorNumeric[T]) : GaussianDropout[T] = {
    new GaussianDropout[T](rate)
  }
}
