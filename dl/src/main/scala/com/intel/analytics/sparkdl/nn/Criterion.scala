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

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang3.SerializationUtils

import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag

class Criterion[@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable {
  var output: T = ev.fromType[Int](0)

  def forward(input: Tensor[T], target: Tensor[T]): T = {
    updateOutput(input, target)
  }

  def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    updateGradInput(input, target)
  }

  def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    this.output
  }

  def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = Tensor[T]()

  def cloneCriterion(): Criterion[T] = {
    SerializationUtils.clone(this)
  }
}
