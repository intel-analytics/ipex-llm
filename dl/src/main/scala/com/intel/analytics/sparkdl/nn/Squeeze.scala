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

class Squeeze[@specialized(Float, Double) T: ClassTag](
  var dim : Int = Int.MinValue,
  var numInputDims: Int = Int.MinValue
  )(implicit ev: TensorNumeric[T]) extends TensorModule[T]  {

  def setNumInputDims(numInputDims: Int): Unit = {
    this.numInputDims = numInputDims
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var addone = false
    if (numInputDims != Int.MinValue && input.dim() == numInputDims + 1) {
      if (dim != Int.MinValue) {
        dim += 1
      } else if (input.size(1) == 1) {
        addone = true // in case of minibatch of size 1
      }
    }
    output.set(input)
    if (dim != Int.MinValue) output.squeeze(dim) else output.squeeze()
//    output.set(if (dim != Int.MinValue) input.squeeze(dim) else input.squeeze())
    if (addone) {
      val s = output.size()
      s(1) = 1
      output.set(output.view(s))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nElement() == gradOutput.nElement())
    gradInput.set(gradOutput.view(input.size()))
    gradInput
  }

  override def toString(): String = {
    s"nn.Squeeze(${if (dim != Int.MinValue) dim + ", " else ""}" +
      s"${if (numInputDims != Int.MinValue) numInputDims else ""})"
  }
}
