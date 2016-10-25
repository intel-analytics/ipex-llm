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

import com.intel.analytics.sparkdl.utils.RandomGenerator._
import scala.reflect.ClassTag

class CAdd[@specialized(Float, Double) T: ClassTag](
  size: Array[Int])(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  val bias: Tensor[T] = Tensor[T](size)
  this.gradBias = Tensor[T](size)
  reset()

  override def reset(): Unit = {
    val stdv = 1.0/math.sqrt(bias.nElement())
    bias.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    if (input.nElement() == bias.nElement()) {
      output.add(bias)
    } else {
      val expand = if (bias.dim() == input.dim()) {
        bias.view(bias.size())
      } else {
        bias.view(Array(1) ++ bias.size())
      }
      expand.expandAs(output)
      output.add(expand)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {

    if (bias.nElement() == gradOutput.nElement()) {
      gradBias.add(ev.fromType[Double](scale), gradOutput)
    } else {
      val expand = if (bias.dim() == gradOutput.dim()) {
        gradBias.view(gradBias.size())
      } else {
        gradBias.view(Array(1) ++ gradBias.size())
      }

      expand.expandAs(gradOutput)
      expand.add(ev.fromType[Double](scale), gradOutput)
    }
  }

  override def updateParameters(learningRate: T): Unit = {
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.bias), Array(this.gradBias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.CAdd($size)"
  }
}
