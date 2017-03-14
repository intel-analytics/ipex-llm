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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Scale is the combination of cmul and cadd
 * Computes the elementwise product of input and weight, with the shape of the weight "expand" to
 * match the shape of the input.
 * Similarly, perform a expand cdd bias and perform an elementwise add
 * @param size size of weight and bias
 * @tparam T Numeric type. Only support float/double now
 */
class Scale[T: ClassTag](val size: Array[Int])
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T] {

  private val cmul = new CMul[T](size)
  private val cadd = new CAdd[T](size)

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   * @param input
   * @return
   */
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = cadd.forward(cmul.forward(input))
    output
  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = cmul.backward(cadd.output, cadd.backward(input, gradOutput))
    gradInput
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(cmul.parameters()._1(0), cadd.parameters()._1(0)),
      Array(cmul.parameters()._2(0), cadd.parameters()._2(0)))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> cmul.weight, "bias" -> cadd.bias,
      "gradWeight" -> cmul.gradWeight, "gradBias" -> cadd.gradBias))
  }

  override def toString: String = "nn.Scale"
}

object Scale {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int])
    (implicit ev: TensorNumeric[T]): Scale[T] = new Scale[T](size)
}
