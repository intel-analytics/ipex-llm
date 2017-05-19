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
 * It is a simple module preserves the input, but takes the
 * gradient from the subsequent layer, multiplies it by -lambda
 * and passes it to the preceding layer. This can be used to maximise
 * an objective function whilst using gradient descent, as described in
 *  ["Domain-Adversarial Training of Neural Networks"
 *  (http://arxiv.org/abs/1505.07818)]
 *
 * @param lambda hyper-parameter lambda can be set dynamically during training
 */

@SerialVersionUID(- 5518750357832311906L)
class GradientReversal[T: ClassTag](var lambda: Double = 1) (implicit ev: TensorNumeric[T])

  extends TensorModule[T] {

  def setLambda(lambda: Double): this.type = {
    this.lambda = lambda
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.set(input)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
      .copy(gradOutput)
      .mul(ev.negative(ev.fromType[Double](lambda)))
  }
}


object GradientReversal {
  def apply[@specialized(Float, Double) T: ClassTag](
      lambda: Double = 1)(implicit ev: TensorNumeric[T]) : GradientReversal[T] = {
    new GradientReversal[T](lambda)
  }
}
