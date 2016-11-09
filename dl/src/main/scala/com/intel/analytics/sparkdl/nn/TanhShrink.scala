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

import scala.reflect.ClassTag

/**
 * A simple layer for each element of the input tensor, do the following operation
 * during the forward process:
 *    [f(x) = tanh(x) - 1]
 */
class TanhShrink[T: ClassTag](
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  private val tanh = new Tanh[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val th = tanh.updateOutput(input)
    output.resizeAs(input).copy(input)
    output.add(ev.fromType[Int](-1), th)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dth = tanh.updateGradInput(input, gradOutput)
    gradInput.resizeAs(input).copy(gradOutput)
    gradInput.add(ev.fromType[Int](-1), dth)
    gradInput
  }

  override def toString: String = s"nn.TanhShrink"
}
