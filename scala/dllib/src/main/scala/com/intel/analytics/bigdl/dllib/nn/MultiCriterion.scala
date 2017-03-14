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

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, AbstractCriterion}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * a weighted sum of other criterions each applied to the same input and target;
 */

@SerialVersionUID(- 8679064077837483164L)
class MultiCriterion[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractCriterion[Activity, Activity, T] {

  private val weights = new ArrayBuffer[Double]
  private val criterions = T()

  def add(criterion: AbstractCriterion[Activity, Activity, T], weight: Double = 1): Unit = {
    criterions.insert(criterions.length() + 1, criterion)
    weights.append(weight)
  }
  override def updateOutput(input: Activity, target: Activity): T = {
    var i = 1
    while (i <= criterions.length) {
      output = ev.plus(output, ev.times(ev.fromType(weights(i-1)),
        criterions[AbstractCriterion[Activity, Activity, T]](i).updateOutput(input, target)))
      i +=1
    }
    output
  }

  override def updateGradInput(input: Activity, target: Activity): Activity = {
    gradInput = Utils.recursiveResizeAs[T](gradInput,
      input)
    Utils.recursiveFill[T](gradInput, 0)
    var i = 1
    while (i <= criterions.length) {
      Utils.recursiveAdd(gradInput, weights(i - 1),
        criterions[AbstractCriterion[Activity, Activity, T]](i).updateGradInput(input, target))
      i += 1
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MultiCriterion[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MultiCriterion[T] =>
      super.equals(that) &&
      (that canEqual this) &&
        weights == that.weights
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), weights)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def toString(): String = {
    s"nn.MultiCriterion"
  }
}

object MultiCriterion {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : MultiCriterion[T] = {
    new MultiCriterion[T]()
  }
}
