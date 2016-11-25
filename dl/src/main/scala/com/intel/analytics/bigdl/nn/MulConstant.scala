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
 * Multiplies input Tensor by a (non-learnable) scalar constant.
 * This module is sometimes useful for debugging purposes.
 * @param scalar scalar constant
 * @param inplace Can optionally do its operation in-place without using extra state memory
 */
class MulConstant[@specialized(Float, Double) T: ClassTag](
  val scalar : T, val inplace : Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T]  {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inplace) {
      input.mul(scalar)
      output.set(input)
    } else {
      output.resizeAs(input).copy(input)
      output.mul(scalar)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (inplace) {
      gradOutput.mul(scalar)
      gradInput.set(gradOutput)
      input.div(scalar)
    } else {
      gradInput.resizeAs(gradOutput).copy(gradOutput)
      gradInput.mul(scalar)
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.MulConstant($scalar, $inplace)"
  }


  override def canEqual(other: Any): Boolean = other.isInstanceOf[MulConstant[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MulConstant[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        scalar == that.scalar &&
        inplace == that.inplace
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), scalar, inplace)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}
