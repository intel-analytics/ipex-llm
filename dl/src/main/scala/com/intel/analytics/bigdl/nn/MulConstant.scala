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

import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

class MulConstant[@specialized(Float, Double) T: ClassTag](
  constantScalar:T,
  ip: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Module[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (ip) {
      input.mul(constantScalar)
      output.set(input)
    } else {
      output.resizeAs(input)
            .copy(input)
            .mul(constantScalar)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (ip) {
      gradOutput.mul(constantScalar)
      gradInput.set(gradOutput)
      input.div(constantScalar)
    } else {
      gradInput = gradInput.resizeAs(gradOutput)
        .copy(gradOutput)
        .mul(constantScalar)
    }
    gradInput
  }
}
