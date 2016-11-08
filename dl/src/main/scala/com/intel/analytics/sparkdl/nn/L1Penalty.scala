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

class L1Penalty[T: ClassTag](l1weight: Int, sizeAverage: Boolean = false,
 provideOutput: Boolean = true)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var loss: T = ev.fromType(0)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var m: Double = l1weight
    if (sizeAverage == true) m = m / input.nElement()
    val tmp = input.abs().sum()
    loss = ev.times(ev.fromType(m), input.abs().sum())
    output = input
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    var m: Double = l1weight

    if (sizeAverage == true) m = m / input.nElement()
    gradInput.resizeAs(input).copy(input).sign().mul(ev.fromType(m))

    if (provideOutput == true) gradInput.add(gradOutput)
    gradInput
  }

  override def toString(): String = {
    s"nn.L1Penalty ($l1weight, $sizeAverage, $provideOutput)"
  }
}
