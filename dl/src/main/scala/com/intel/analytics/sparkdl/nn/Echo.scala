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
 * This module is for debug purpose, which can print activation and gradient in your model
 * topology
 * @param ev$1
 * @param ev
 * @tparam T
 */
class Echo[@specialized(Float, Double) T: ClassTag] (implicit ev: TensorNumeric[T])
  extends Module[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output = input
    println(s"${getName()} : Activation size is ${input.size().mkString("x")}")
    this.output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput
    println(s"${getName()} : Gradient size is ${gradOutput.size().mkString("x")}")
    this.gradInput
  }

  override def toString(): String = {
    s"nn.Echo"
  }
}
