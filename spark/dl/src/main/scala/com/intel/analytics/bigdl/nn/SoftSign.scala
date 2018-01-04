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
 * Apply SoftSign function to an n-dimensional input Tensor.
 *
 * SoftSign function: f_i(x) = x_i / (1+|x_i|)
 */

@SerialVersionUID(- 3936698382129844874L)
class SoftSign[T: ClassTag, D: ClassTag]()
    (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T] {

  @transient private var temp: Tensor[D] = null
  @transient private var tempGrad: Tensor[D] = null

  output = Tensor[D]()
  gradInput = Tensor[D]()

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    if (null == temp) {
      temp = input.clone()
    } else {
      temp.resizeAs(input).copy(input)
    }
    temp.abs().add(ev2.fromType[Int](1))
    output.resizeAs(input).copy(input).cdiv(temp)
    output
  }

  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    if (null == tempGrad) {
      tempGrad = input.clone()
    } else {
      tempGrad.resizeAs(output).copy(input)
    }
    tempGrad.abs().add(ev2.fromType[Int](1)).cmul(tempGrad)
    gradInput.resizeAs(input).copy(gradOutput).cdiv(tempGrad)
    gradInput
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object SoftSign {
  def apply[@specialized(Float, Double) T: ClassTag, D: ClassTag]()
      (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]) : SoftSign[T, D] = {
    new SoftSign[T, D]()
  }
}
