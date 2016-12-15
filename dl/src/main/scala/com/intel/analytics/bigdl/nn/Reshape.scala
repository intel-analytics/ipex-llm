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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Reshape[@specialized(Float, Double) T: ClassTag](
  size: Array[Int], var batchMode: Option[Boolean] = None)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T]  {
  val batchSize = new Array[Int](size.length + 1)
  var nElement: Int = 1
  for (i <- 1 to size.length) {
    batchSize(i) = size(i - 1)
    nElement *= size(i - 1)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {

    if ((batchMode.nonEmpty && batchMode.get == false) ||
      (input.nElement() == nElement && batchMode.isEmpty && input.size(1) != 1)) {
      require(input.nElement() == nElement, "element number must match Reshape size")
      if (input.isContiguous()) output =
        input.view(size)
      else output = input.contiguous().view(size)
    }
    else {
      require(input.nElement() == nElement * input.size(1),
        "element number must match Reshape size")
      batchSize(0) = input.size(1)
      if (input.isContiguous()) {
        output = input.view(batchSize)
      } else {
        output = input.contiguous().view(batchSize)
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradOutput.isContiguous()) {
      gradInput = gradOutput.view(input.size())
    } else {
      gradInput = gradOutput.contiguous().view(input.size())
    }
    gradInput
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Reshape[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Reshape[T]]
    if (this.eq(other)) {
      return true
    }

    var i = 0
    while (i < batchSize.length) {
      if (batchSize(i) != other.batchSize(i)) {
        return false
      }
      i += 1
    }
    nElement == other.nElement &&
      batchMode == other.batchMode
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    var i = 0
    while (i < batchSize.length) {
      hash = hash * seed + batchSize(i).hashCode()
      i += 1
    }
    hash = hash * seed + nElement.hashCode()
    hash = hash * seed + batchMode.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.Reshape(${size.mkString("x")})"
  }
}

object Reshape {
  def apply[@specialized(Float, Double) T: ClassTag](
      size: Array[Int],
      batchMode: Option[Boolean] = None)(implicit ev: TensorNumeric[T]) : Reshape[T] = {
    new Reshape[T](size, batchMode)
  }
}
