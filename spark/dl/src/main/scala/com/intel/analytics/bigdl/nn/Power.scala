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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Apply an element-wise power operation with scale and shift.
 *
 * f(x) = (shift + scale * x)^power^
 *
 * @param power the exponent.
 * @param scale Default is 1.
 * @param shift Default is 0.
 */

@SerialVersionUID(- 6637789603381436472L)
class Power[T: ClassTag, D: ClassTag](
  val power: Double,
  val scale : Double = 1,
  val shift : Double = 0)
(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T] {

  output = Tensor[D]()
  gradInput = Tensor[D]()

  val diffScale = power * scale

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    output.resizeAs(input)
    output.copy(input)
    if(scale != 1) {
      output.mul(ev2.fromType[Double](scale))
    }
    if(shift != 0) {
      output.add(ev2.fromType[Double](shift))
    }
    if(power != 1) {
      output.pow(output, ev2.fromType[Double](power))
    }

    output
  }


  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    gradInput.resizeAs(input)
    // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
    //               = diff_scale * y / (shift + scale * x)
    if(power == 2) {
      // Special case for y = (shift + scale * x)^2
      //     -> dy/dx = 2 * scale * (shift + scale * x)
      //              = diff_scale * shift + diff_scale * scale * x
      gradInput.copy(input)
      gradInput.mul(ev2.fromType[Double](diffScale * scale))
      if(shift != 0) {
        gradInput.add(ev2.fromType(diffScale * shift))
      }
    } else if (shift == 0) {
      // Special case for y = (scale * x)^power
      //     -> dy/dx = scale * power * (scale * x)^(power - 1)
      //              = scale * power * (scale * x)^power * (scale * x)^(-1)
      //              = power * y / x
      gradInput.fill(ev2.fromType[Int](0))
      gradInput.addcdiv(ev2.fromType[Double](power), output, input)
    } else {
      gradInput.copy(input)
      if(scale != 1) {
        gradInput.mul(ev2.fromType[Double](scale))
      }
      if(shift != 0) {
        gradInput.add(ev2.fromType[Double](shift))
      }
      gradInput.cdiv(output, gradInput)
      if (diffScale != 1) {
        gradInput.mul(ev2.fromType[Double](diffScale))
      }
    }
    if(diffScale != 0) {
      gradInput.cmul(gradOutput)
    }

    gradInput
  }

  override def toString(): String = {
    s"${getPrintName}($power, $scale, $shift)"
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }

}

object Power {
  def apply[@specialized(Float, Double) T: ClassTag, D: ClassTag](
      power: Double,
      scale : Double = 1,
      shift : Double = 0)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Power[T, D] = {
    new Power[T, D](power, scale, shift)
  }
}
