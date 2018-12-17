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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

/**
 * The mean squared error criterion
 * e.g. input: a, target: b, total elements: n
 * loss(a, b) = 1/n \sum |a_i - b_i|^2
 * sizeAverage is true by default to divide the sum of squared error by n
 */
@SerialVersionUID(- 7078521754128606735L)
class MSECriterion[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  var sizeAverage = true

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    gradInput.resizeAs(input).copy(input)
    gradInput.sub(target)
    output = gradInput.dot(gradInput)
    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    var norm = ev.fromType[Int](2)
    if (sizeAverage) {
      norm = ev.fromType[Double](2.0 / input.nElement)
    }
    gradInput.mul(norm)
    gradInput
  }

}

object MSECriterion {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : MSECriterion[T] = {
    new MSECriterion[T]()
  }
}
