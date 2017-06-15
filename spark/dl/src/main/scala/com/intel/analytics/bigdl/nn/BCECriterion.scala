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
(var weights: Tensor[T] = null, sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  private val eps = 1e-12
  if (weights != null) require(weights.dim() == 1, "weights input should be 1-D Tensor")

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.nElement() == target.nElement())

    if (null != weights && target.dim() != 1) {
      weights = weights.view(1, target.size(2)).expandAs(target)
    }

    var sum = 0.0
    if (null != weights) {
      val func = new TensorFunc6[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                           data3: Array[T], offset3: Int): Unit = {
          val x = ev.toType[Double](data1(offset1))
          val y = ev.toType[Double](data2(offset2))
          val w = ev.toType[Double](data3(offset3))
          sum -= (Math.log(x + eps) * y + Math.log(1.0 - x + eps) * (1.0 - y)) * w
        }
      }
      DenseTensorApply.apply3(input, target, weights, func)
    } else {
      val func = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int,
                           data2: Array[T], offset2: Int): Unit = {
          val x = ev.toType[Double](data1(offset1))
          val y = ev.toType[Double](data2(offset2))
          sum -= Math.log(x + eps) * y + Math.log(1.0 - x + eps) * (1.0 - y)
        }
      }
      DenseTensorApply.apply2(input, target, func)

    }

    if (sizeAverage) sum /= input.nElement()

    output = ev.fromType[Double](sum)

    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == target.nElement())

    if (null != weights && target.dim() != 1) {
      weights = weights.view(1, target.size(2)).expandAs(target)
    }

    val norm = if (sizeAverage) 1.0 / input.nElement() else 1.0

    gradInput.resizeAs(input)

    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                         data3: Array[T], offset3: Int): Unit = {
        val x = ev.toType[Double](data2(offset2))
        val y = ev.toType[Double](data3(offset3))
        data1(offset1) = ev.fromType(-norm * (y - x) / ((1.0 - x + eps) * (x + eps)))
      }
    }
    DenseTensorApply.apply3(gradInput, input, target, func)

    if (null != weights) {
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
