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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc4, TensorFunc6}

import scala.reflect.ClassTag

/**
 * This loss function measures the Binary Cross Entropy between the target and the output
 *         loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
 * or in the case of the weights argument being specified:
 *         loss(o, t) = - 1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
 *
 * By default, the losses are averaged for each mini-batch over observations as well as over
 * dimensions. However, if the field sizeAverage is set to false, the losses are instead summed.
 * @param weights weights over the input dimension
 * @param sizeAverage avgerage or not in each mini-batch
 * @param ev numeric operator
 * @tparam T numeric type
 */
@SerialVersionUID(- 1953992758534446600L)
class BCECriterion[@specialized(Float, Double) T: ClassTag]
(val weights: Tensor[T] = null, sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  private val eps = 1e-12

  val buffer: Tensor[T] = Tensor[T]()

  val onesBuffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.size().sameElements(target.size()),
      s"input size should be equal to target size, but got input size: ${input.size().toList}," +
        s" target size: ${target.size().toList}")

    if (weights != null) {
      if (weights.nDimension() < input.nDimension()) {
        require(weights.size().sameElements(input.size().tail),
          s"weights size should be equal to input size or input size's tail, but got" +
            s" input size: ${input.size().toList}, weights size: ${weights.size().toList}")
      } else if (weights.nDimension() == input.nDimension()) {
        require(weights.size().sameElements(input.size()),
          s"weights size should be equal to input size or input size's tail, but got" +
            s" input size: ${input.size().toList}, weights size: ${weights.size().toList}")
      } else {
        throw new IllegalArgumentException(
          s"weights size should be equal to input size or input size's tail, but got" +
          s" input size: ${input.size().toList}, weights size: ${weights.size().toList}")
      }
    }

    var sum = 0.0
    if (null != weights) {
      buffer.resizeAs(input).copy(input).add(ev.fromType(eps)).log()
      // cmul support broadcasting
      buffer.cmul(weights)
      sum += ev.toType[Double](buffer.dot(target))
      buffer.fill(ev.one).sub(input).add(ev.fromType(eps)).log().cmul(weights)
      sum -= ev.toType[Double](buffer.dot(target))
      if (onesBuffer.nElement() != buffer.nElement()) {
        onesBuffer.resizeAs(buffer).fill(ev.one)
      }
      sum += ev.toType[Double](buffer.dot(onesBuffer))
    } else {
      buffer.resizeAs(input).copy(input).add(ev.fromType(eps)).log()
      sum += ev.toType[Double](buffer.dot(target))
      buffer.fill(ev.one).sub(input).add(ev.fromType(eps)).log()
      sum -= ev.toType[Double](buffer.dot(target))
      if (onesBuffer.nElement() != buffer.nElement()) {
        onesBuffer.resizeAs(buffer).fill(ev.one)
      }
      sum += ev.toType[Double](buffer.sum())
    }

    if (sizeAverage) sum /= input.nElement()

    output = ev.fromType[Double](-sum)

    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == target.nElement(),
      "input and target should have the same dims." +
        s"input dim(${input.nElement()})" +
        s"target dim(${target.nElement()})")

    val nElement = input.nElement()
    val norm = if (sizeAverage) 1.0 / nElement else 1.0

    gradInput.resizeAs(input)

    // gradInput = -norm * (y - x) / ((1.0 - x + eps) * (x + eps))
    // - (1 - x + eps)*(x + eps) = x^2 - x - eps - eps^2
    // eps^12 is negligible
    buffer.pow(input, ev.fromType(2)).sub(input).sub(ev.fromType(eps))
    gradInput.copy(target).sub(input).cdiv(buffer).mul(ev.fromType(norm))

    if (null != weights) {
      // cmul support broadcasting
      gradInput.cmul(weights)
    }

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
