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


class AbsCriterion[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true)
(implicit ev: TensorNumeric[T]) extends Criterion[T] {

  var buffer = Tensor[T]()
  var gradInput: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target : Tensor[T]): T = {
    buffer.resizeAs(input).add(input)
    buffer.mul(input, ev.fromType[Int](-1)).add(target).abs()

    output = buffer.sum()
    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    var norm : Double = 0
    if (sizeAverage)  {
      norm = 1.0/input.nElement()
    } else {
      norm = 1.0
    }
    gradInput.mul(input, ev.fromType[Int](-1)).add(target)

    val bufferArray = gradInput.storage().array()
    val bufferOffset = gradInput.storageOffset() - 1
    var i = 0
    while(i < gradInput.nElement()) {
      val z = bufferArray(i)
      bufferArray(i + bufferOffset) = ev.times(ev.fromType(norm),
        if (ev.isGreater(z, ev.fromType(0))) ev.fromType(-1) else ev.fromType(1))
      i += 1
    }
    gradInput
  }
}
