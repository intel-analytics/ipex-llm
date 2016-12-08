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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.commons.lang3.SerializationUtils

import scala.reflect.ClassTag

abstract class TensorCriterion[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends Criterion[Tensor[T], T]

abstract class Criterion[A <: Activity: ClassTag,
@specialized(Float, Double) T: ClassTag](
  implicit ev: TensorNumeric[T]) extends Serializable {
  @transient var gradInput: A = Activity[A, T]()
  var output: T = ev.fromType[Int](0)

  private[nn] def allocateAs[D <: Activity](dest: D): D = dest match {
    case tensor: Tensor[T] => Tensor[T]().asInstanceOf[D]
    case table: Table => T().asInstanceOf[D]
    case _ => throw new IllegalArgumentException("Activity only support tensor and table now")
  }

  def forward(input: A, target: A): T = {
    updateOutput(input, target)
  }

  def backward(input: A, target: A): A = {
    updateGradInput(input, target)
  }

  def updateOutput(input: A, target: A): T = {
    this.output
  }

  def updateGradInput(input: A, target: A): A

  def cloneCriterion(): Criterion[A, T] = {
    SerializationUtils.clone(this)
  }


  def canEqual(other: Any): Boolean = other.isInstanceOf[Criterion[A, T]]

  override def equals(other: Any): Boolean = other match {
    case that: Criterion[A, T] =>
      (that canEqual this) &&
        (that.getClass equals this.getClass) &&
        output == that.output
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(output)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}
