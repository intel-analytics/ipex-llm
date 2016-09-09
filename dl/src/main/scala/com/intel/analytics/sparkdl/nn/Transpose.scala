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

class Transpose[@specialized(Float, Double) T: ClassTag](
  val permutations: Array[(Int, Int)])(implicit ev: TensorNumeric[T]) extends Module[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    var i = 0
    while (i < permutations.length) {
      output.transpose(permutations(i)._1, permutations(i)._2)
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput).copy(gradOutput)
    var i = permutations.length - 1
    while (i >= 0) {
      gradInput.transpose(permutations(i)._1, permutations(i)._2)
      i -= 1
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.Transpose(${
      permutations.map {
        case (from: Int, to: Int) => s"$from -> $to"
      }.mkString(", ")
    })"
  }
}
