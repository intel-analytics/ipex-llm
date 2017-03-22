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
 * Creates a criterion that measures the Binary Cross Entropy
 * between the target and the output
 *
 * @param weights weights for each class
 * @param sizeAverage whether to average the loss or not
 */

@SerialVersionUID(- 1953992758534446600L)
class BCECriterion[@specialized(Float, Double) T: ClassTag]
(var weights: Tensor[T] = null, sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  var total_weight = ev.fromType[Int](0)
  val eps = ev.fromType[Double](1e-12)
  if (weights != null) require(weights.dim() == 1, "weights input should be 1-D Tensor")

  var buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.nElement() == target.nElement())
    buffer.resizeAs(input).zero()

    if (null != weights && target.dim() != 1) {
      weights = weights.view(1, target.size(2)).expandAs(target)
    }

    buffer.add(input).add(eps)
    buffer.apply1(ev.log(_))

    if (null != weights) buffer.cmul(weights)

    output = target.dot(buffer)

    buffer.mul(input, ev.fromType[Int](-1)).add(ev.fromType[Int](1)).add(eps).apply1(ev.log)
    if (null != weights) buffer.cmul(weights)

    output = ev.plus(output, buffer.sum())
    output = ev.minus(output, target.dot(buffer))

    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))

    output = ev.negative(output)

    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == target.nElement())

    if (null != weights && target.dim() != 1) {
      weights = weights.view(1, target.size(2)).expandAs(target)
    }

    buffer.resizeAs(input)
    buffer.zero()
    // -x ( 1 + eps - x) + eps
    buffer.add(ev.fromType[Int](-1)).add(input).add(ev.negative(eps)).
      cmul(input).add(ev.negative(eps))

    gradInput.resizeAs(input)
    // y - x
    gradInput.add(target, ev.fromType[Int](-1), input)
    // - (y - x) / ( x ( 1 + eps -x ) + eps )
    gradInput = gradInput / buffer

    if (null != weights) gradInput.cmul(weights)

    if (sizeAverage) gradInput.div(ev.fromType[Int](target.nElement()))

    gradInput
  }
}


object BCECriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    weights: Tensor[T] = null,
    sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : BCECriterion[T] = {
    new BCECriterion[T](weights, sizeAverage)
  }
}
