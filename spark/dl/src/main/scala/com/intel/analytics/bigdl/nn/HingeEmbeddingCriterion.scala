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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}

import scala.reflect.ClassTag

/**
 * Creates a criterion that measures the loss given an
 * input x which is a 1-dimensional vector and a label y (1 or -1).
 * This is usually used for measuring whether two inputs are similar
 * or dissimilar,
 * e.g. using the L1 pairwise distance, and is typically used for
 * learning nonlinear embeddings or semi-supervised learning.

                    *⎧ x_i,                  if y_i ==  1
   *loss(x, y) = 1/n ⎨
                    *⎩ max(0, margin - x_i), if y_i == -1

 * If x and y are n-dimensional Tensors, the sum operation still operates
 * over all the elements, and divides by n (this can be avoided if one sets
 * the internal variable sizeAverage to false). The margin has a default
 * value of 1, or can be set in the constructor.
 *
 * @param margin
 * @param sizeAverage
 */

@SerialVersionUID(117094129660790270L)
class HingeEmbeddingCriterion[@specialized(Float, Double) T: ClassTag](
  margin: Double = 1,
  sizeAverage: Boolean = true
)(implicit ev: TensorNumeric[T])
  extends TensorCriterion[T] {
  @transient private var buffer: Tensor[T] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    if (buffer == null) {
      buffer = Tensor[T]()
    }

    buffer.resizeAs(input).copy(input)
    buffer.map(target, (bu, y) => {
      if (y == ev.fromType[Int](-1)) {
        ev.fromType[Int](0)
      } else {
        bu
      }
    })
    output = buffer.sum

    buffer.fill(ev.fromType[Double](margin))
      .add(ev.fromType[Int](-1), input)
    buffer.cmax(ev.fromType[Int](0))
    buffer.map(target, (bu, in) => {
      if (in == ev.fromType[Int](1)) {
        ev.fromType[Int](0)
      } else {
        bu
      }
    })
    output = ev.plus(output, buffer.sum())

    if (sizeAverage) {
      output = ev.divide(output, ev.fromType[Int](input.nElement()))
    }

    output
  }

  // TODO: Optimize performance to substitute apply3
  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).copy(target)
    val func = new TensorFunc6[T] {
      override def apply (data1: Array[T], offset1: Int, data2: Array[T],
        offset2: Int, data3: Array[T], offset3: Int): Unit = {
        if (ev.fromType[Int](-1) == data2(offset2) &&
          ev.isGreater(data3(offset3), ev.fromType[Double](margin))) {
          data1(offset1) = ev.fromType[Int](0)
        } else {
          data1(offset1) = data1(offset1)
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, target, input, func)

    if (sizeAverage) {
      gradInput.mul(ev.fromType[Double](1.0 / input.nElement()))
    }

    gradInput
  }

  override def toString: String = s"nn.HingeEmbeddingCriterion"
}

object HingeEmbeddingCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      margin: Double = 1,
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : HingeEmbeddingCriterion[T] = {
    new HingeEmbeddingCriterion[T](margin, sizeAverage)
  }
}
