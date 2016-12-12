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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Activities, T}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * a weighted sum of other criterions each applied to the same input and target;
 */
class MultiCriterion[A <: Activities: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends Criterion[A, T] {

  private var gradInput: A = Activities[A, T]().asInstanceOf[A]
  private val weights = new ArrayBuffer[Double]
  private val criterions = T()

  def add(criterion: Criterion[A, T], weight: Double = 1): Unit = {
    criterions.insert(criterions.length() + 1, criterion)
    weights.append(weight)
  }
  override def updateOutput(input: A, target: A): T = {
    var i = 1
    while (i <= criterions.length) {
      output = ev.plus(output, ev.times(ev.fromType(weights(i-1)),
        criterions[Criterion[A, T]](i).updateOutput(input, target)))
      i +=1
    }
    output
  }

  override def updateGradInput(input: A, target: A): A = {
    gradInput = Utils.recursiveResizeAs[T](gradInput.asInstanceOf[Activities],
      input).asInstanceOf[A]
    Utils.recursiveFill[T](gradInput, 0)
    var i = 1
    while (i <= criterions.length) {
      Utils.recursiveAdd(gradInput, weights(i - 1),
        criterions[Criterion[A, T]](i).updateGradInput(input, target))
      i += 1
    }
    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MultiCriterion[A, T]]

  override def equals(other: Any): Boolean = other match {
    case that: MultiCriterion[A, T] =>
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
  def apply[A <: Activities: ClassTag, @specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : MultiCriterion[A, T] = {
    new MultiCriterion[A, T]()
  }
}
