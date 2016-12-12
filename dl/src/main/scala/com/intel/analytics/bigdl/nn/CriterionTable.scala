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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Creates a module that wraps a Criterion so that it can accept a table of inputs.
 *
 * @param criterion Criterion module
 */
class CriterionTable[T: ClassTag](val criterion: TensorCriterion[T])
 (implicit ev: TensorNumeric[T]) extends  TensorCriterion[T] {

  @transient
  var gradInput: Tensor[T] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    output = criterion.updateOutput(input, target)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    if (null == gradInput) gradInput = Tensor[T]()
    gradInput = criterion.updateGradInput(input, target)
    gradInput
  }

  override def toString(): String = {
    s"nn.CriterionTable"
  }
}

object CriterionTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      criterion: TensorCriterion[T])(implicit ev: TensorNumeric[T]) : CriterionTable[T] = {
    new CriterionTable[T](criterion)
  }
}
