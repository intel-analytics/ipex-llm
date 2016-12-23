/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 *  adding a constant
 *
 * @param constant_scalar constant value
 * @param inplace Can optionally do its operation in-place without using extra state memory
 */
@SerialVersionUID(- 1572711921601326233L)
class AddConstant[T: ClassTag](
   val constant_scalar: T,
   val inplace: Boolean = false
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inplace) {
      input.add(constant_scalar)
      output.set(input)
    } else {
      output.resizeAs(input).copy(input)
      output.add(constant_scalar)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (inplace) {
      gradInput.set(gradOutput)
      input.add(ev.negative(constant_scalar))
    } else {
      gradInput.resizeAs(input).copy(gradOutput)
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.AddConstant ($constant_scalar, $inplace)"
  }

}

object AddConstant {
  def apply[@specialized(Float, Double) T: ClassTag](
    constant_scalar: T,
    inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : AddConstant[T] = {
    new AddConstant[T](constant_scalar, inplace)
  }
}
