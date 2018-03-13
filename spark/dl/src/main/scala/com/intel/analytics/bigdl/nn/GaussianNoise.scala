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
 * Apply additive zero-centered Gaussian noise.
 * This is useful to mitigate overfitting (you could see it as a form of random data
 * augmentation).
 * Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.
 * As it is a regularization layer, it is only active at training time.
 *
 * Output shape is the same as input.
 *
 *
 * @param stddev double, standard deviation of the noise distribution.
 *
 */

@SerialVersionUID(- 2590701089601246637L)
class GaussianNoise[T: ClassTag](
   val stddev: Double
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T] {


  override def updateOutput(input: Tensor[T]): Tensor[T] = {

    this.output.resizeAs(input).copy(input)

    if(train) {
      val noise = Tensor[T]()
      noise.resizeAs(input)
      noise.randn(0.0, stddev)
      this.output.add(noise)
    } else {
      this.output
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    if (train) {
      this.gradInput.resizeAs(gradOutput).copy(gradOutput)
    } else {
      throw new IllegalArgumentException("backprop only defined while training")
    }
    this.gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($stddev)"
  }


}

object GaussianNoise {
  def apply[@specialized(Float, Double) T: ClassTag](
    stddev: Double
    )(implicit ev: TensorNumeric[T]) : GaussianNoise[T] = {
    new GaussianNoise[T](stddev)
  }
}
