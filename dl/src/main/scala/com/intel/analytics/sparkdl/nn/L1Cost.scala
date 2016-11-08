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

import com.intel.analytics.sparkdl.tensor.{DenseTensorApply, Tensor, TensorFunc4}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class L1Cost[T: ClassTag]()
 (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  val gradInput = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    input.abs().sum()
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    val func = new TensorFunc4[T] {
      override def apply(data1: Array[T], index1: Int, data2: Array[T], index2: Int): Unit = {
        if (ev.isGreater(data1(index1), ev.fromType(0))) {
          data2(index2) = ev.fromType(1)
        } else if (ev.isGreater(ev.fromType(0), data1(index1))) {
          data2(index2) = ev.fromType(-1)
        } else {
          data2(index2) = ev.fromType(0)
        }
      }
    }
    DenseTensorApply.apply2[T](input, gradInput, func)
    gradInput
  }

  override def toString(): String = {
    s"nn.L1Cost"
  }
}
