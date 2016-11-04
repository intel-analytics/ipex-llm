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
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag


class CrossEntropyCriterion[T: ClassTag](var weights: Tensor[T] = null)
                                        (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  var gradInput: Tensor[T] = Tensor[T]()

  val nll = new ClassNLLCriterion(weights)
  val lsm = new LogSoftMax()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    input.squeeze()
    target.squeeze()
    lsm.updateOutput(input)
    nll.updateOutput(lsm.output, target)
    output = nll.output
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val size = input.size
    input.squeeze()
    target.squeeze()
    nll.updateGradInput(lsm.output, target)
    lsm.updateGradInput(input, nll.gradInput)
    this.gradInput = lsm.gradInput.view(size)
    this.gradInput

  }
  override def toString(): String = {
    s"nn.CrossEntropyCriterion"
  }
}
