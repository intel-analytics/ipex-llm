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

import com.intel.analytics.bigdl.nn.abstractnn.{IdentityOutputShape, TensorModule}
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc4, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Apply the SoftPlus function to an n-dimensional input tensor.
 *
 * SoftPlus function: f_i(x) = 1/beta * log(1 + exp(beta * x_i))
 *
 * @param beta Controls sharpness of transfer function
 */

@SerialVersionUID(- 6938956677043843473L)
class SoftPlus[T: ClassTag](
    val beta: Double = 1.0
  )( implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  // Avoid floating point issues with exp(x), x>20
  private val threshold = ev.fromType[Double](20.0)
  private val betaT = ev.fromType[Double](beta)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    // f(x) = 1/beta * log(1 + exp(beta * x))
    val func = new TensorFunc4[T] {
      override def apply (data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
        data1(offset1) = if (ev.isGreater(ev.times(data2(offset2), betaT), threshold)) {
          data2(offset2)
        } else {
          ev.divide(ev.log1p(ev.exp(ev.times(data2(offset2), betaT))), betaT)
        }
      }
    }
    DenseTensorApply.apply2[T](output, input, func)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    // d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
    // SINCE
    // y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
    // THEREFORE:
    // d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y)
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
                         data3: Array[T], offset3: Int): Unit = {
        val z = ev.exp(ev.times(data3(offset3), betaT))
        data1(offset1) = if (ev.isGreater(ev.times(data3(offset3), betaT), threshold)) {
          data2(offset2)
        } else {
          ev.times(data2(offset2), ev.divide(ev.minus(z, ev.fromType[Int](1)), z))
        }
      }
    }
    DenseTensorApply.apply3[T](gradInput, gradOutput, output, func)

    gradInput
  }

}

object SoftPlus {
  def apply[@specialized(Float, Double) T: ClassTag](
      beta: Double = 1.0)
      (implicit ev: TensorNumeric[T]) : SoftPlus[T] = {
    new SoftPlus[T](beta)
  }
}
