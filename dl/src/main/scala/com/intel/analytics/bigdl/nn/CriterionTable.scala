/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Creates a module that wraps a Criterion so that it can accept a table of inputs.
 *
 * @param criterion Criterion module
 */

@SerialVersionUID(- 4523512241932683396L)
class CriterionTable[@specialized(Float, Double) T: ClassTag]
(val criterion: TensorCriterion[T])
 (implicit ev: TensorNumeric[T]) extends  TensorCriterion[T] {

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

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CriterionTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: CriterionTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        criterion == that.criterion
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), criterion)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object CriterionTable {
  def apply[@specialized(Float, Double) T: ClassTag](
      criterion: TensorCriterion[T])(implicit ev: TensorNumeric[T]) : CriterionTable[T] = {
    new CriterionTable[T](criterion)
  }
}
