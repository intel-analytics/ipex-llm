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

/**
 * multiply a single scalar factor to the incoming data
 */
class Mul[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var weight = Tensor[T](1)
  this.gradWeight = Tensor[T](1)

  reset()

  override def reset(): Unit = {
    val stdv = 1 / math.sqrt(weight.size(1))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    val tmp = weight.storage().array()
    output.mul(tmp(0))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val tmp = weight.storage().array()
    gradInput.add(tmp(0), gradOutput)
    gradInput
  }


  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
  scale: Double = 1.0): Unit = {
    gradWeight.add(ev.times(input.dot(gradOutput), ev.fromType(scale)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def toString(): String = {
    s"nn.Mul"
  }
}
