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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc4, TensorFunc6}

import scala.reflect.ClassTag

/**
 * Apply the soft shrinkage function element-wise to the input Tensor
 *
 * SoftShrinkage operator:
 *        ⎧ x - lambda, if x >  lambda
 * f(x) = ⎨ x + lambda, if x < -lambda
 *        ⎩ 0, otherwise
 *
 * @param lambda Default is 0.5.
 */

@SerialVersionUID(- 2868096135424517459L)
class SoftShrink[T: ClassTag](
    val lambda: Double = 0.5
  )( implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    val func = new TensorFunc4[T] {
      override def apply (data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        data1(offset1) = if (ev.toType[Double](data2(offset2)) > lambda) {
          ev.minus(data2(offset2), ev.fromType[Double](lambda))
        } else if (ev.toType[Double](data2(offset2)) < - lambda) {
          ev.plus(data2(offset2), ev.fromType[Double](lambda))
        } else {
          ev.fromType[Int](0)
        }
      }
    }
    DenseTensorApply.apply2[T](output, input, func)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                         data3: Array[T], offset3: Int): Unit = {
        data1(offset1) = if (ev.toType[Double](data3(offset3)) > lambda ||
        ev.toType[Double](data3(offset3)) < - lambda) {
          data2(offset2)
        } else {
          ev.fromType[Int](0)
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, gradOutput, input, func)

    gradInput
  }
}

object SoftShrink {
  def apply[@specialized(Float, Double) T: ClassTag](
      lambda: Double = 0.5)(implicit ev: TensorNumeric[T]) : SoftShrink[T] = {
    new SoftShrink[T](lambda)
  }
}
