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
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.RandomGenerator._

import scala.reflect.ClassTag

class SoftSign[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient private var temp: Tensor[T] = null
  @transient private var tempGrad: Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (null == temp) {
      temp = input.clone()
    } else {
      temp.resizeAs(input).copy(input)
    }
    temp.abs().add(ev.fromType[Int](1))
    output.resizeAs(input).copy(input).cdiv(temp)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (null == tempGrad) {
      tempGrad = input.clone()
    } else {
      tempGrad.resizeAs(output).copy(input)
    }
    tempGrad.abs().add(ev.fromType[Int](1)).cmul(tempGrad)
    gradInput.resizeAs(input).copy(gradOutput).cdiv(tempGrad)
    gradInput
  }

  override def toString(): String = {
    s"nn.SoftSign"
  }
}
