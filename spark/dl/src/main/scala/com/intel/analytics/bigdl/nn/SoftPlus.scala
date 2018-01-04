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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
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
class SoftPlus[T: ClassTag, D: ClassTag](
    val beta: Double = 1.0
  )( implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T] {

  // Avoid floating point issues with exp(x), x>20
  private val threshold = ev2.fromType[Double](20.0)
  private val betaT = ev2.fromType[Double](beta)

  output = Tensor[D]()
  gradInput = Tensor[D]()

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    output.resizeAs(input)

    // f(x) = 1/beta * log(1 + exp(beta * x))
    val func = new TensorFunc4[D] {
      override def apply (data1: Array[D], offset1: Int, data2: Array[D], offset2: Int): Unit = {
        data1(offset1) = if (ev2.isGreater(ev2.times(data2(offset2), betaT), threshold)) {
          data2(offset2)
        } else {
          ev2.divide(ev2.log1p(ev2.exp(ev2.times(data2(offset2), betaT))), betaT)
        }
      }
    }
    DenseTensorApply.apply2[D](output, input, func)

    output
  }

  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    gradInput.resizeAs(input)

    // d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
    // SINCE
    // y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
    // THEREFORE:
    // d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y)
    val func = new TensorFunc6[D] {
      override def apply(data1: Array[D], offset1: Int, data2: Array[D], offset2: Int,
                         data3: Array[D], offset3: Int): Unit = {
        val z = ev2.exp(ev2.times(data3(offset3), betaT))
        data1(offset1) = if (ev2.isGreater(ev2.times(data3(offset3), betaT), threshold)) {
          data2(offset2)
        } else {
          ev2.times(data2(offset2), ev2.divide(ev2.minus(z, ev2.fromType[Int](1)), z))
        }
      }
    }
    DenseTensorApply.apply3[D](gradInput, gradOutput, output, func)

    gradInput
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object SoftPlus {
  def apply[@specialized(Float, Double) T: ClassTag, D: ClassTag](
      beta: Double = 1.0)
      (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) : SoftPlus[T, D] = {
    new SoftPlus[T, D](beta)
  }
}
