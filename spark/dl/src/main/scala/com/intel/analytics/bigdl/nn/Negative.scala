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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}

import scala.reflect.ClassTag

/**
 * Computing negative value of each element of input tensor
 * @param inplace output tensor reuse input tensor storage, default is false
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class Negative[T: ClassTag](inplace : Boolean = false)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[_], Tensor[_], T] {

  override def updateOutput(input: Tensor[_]): Tensor[_] = {
    if (inplace) {
      output = input
    } else {
      if (output.getType() != input.getType()) {
        output = input.emptyInstance()
      }
      output.resizeAs(input)
    }

    output.asInstanceOf[Tensor[NumericWildCard]]
      .negative(input.asInstanceOf[Tensor[NumericWildCard]])
  }

  override def updateGradInput(input: Tensor[_], gradOutput: Tensor[_]): Tensor[_] = {
    if (inplace) {
      gradInput = gradOutput
    } else {
      if (gradInput.getType() != gradOutput.getType()) {
        gradInput = gradOutput.emptyInstance()
      }
      gradInput.resizeAs(gradOutput)
    }

    gradInput.asInstanceOf[Tensor[NumericWildCard]]
      .negative(gradOutput.asInstanceOf[Tensor[NumericWildCard]])
  }
}

object Negative {
  def apply[T: ClassTag](inplace: Boolean = false)
    (implicit ev: TensorNumeric[T]): Negative[T] = new Negative[T](inplace)
}
