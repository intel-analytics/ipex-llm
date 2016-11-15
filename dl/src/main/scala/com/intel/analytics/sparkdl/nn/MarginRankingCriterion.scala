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
import com.intel.analytics.sparkdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Creates a criterion that measures the loss given an input x = {x1, x2},
 * a table of two Tensors of size 1 (they contain only scalars), and a label y (1 or -1).
 * In batch mode, x is a table of two Tensors of size batchsize, and y is a Tensor of size
 * batchsize containing 1 or -1 for each corresponding pair of elements in the input Tensor.
 * If y == 1 then it assumed the first input should be ranked higher (have a larger value) than
 * the second input, and vice-versa for y == -1.
 * @param margin
 */
class MarginRankingCriterion[T: ClassTag](margin: Double = 1.0)
 (implicit ev: TensorNumeric[T]) extends Criterion[Table, T] {
  val sizeAverage = true
  val gradInput = T()

  override def updateOutput(input: Table, y: Table): T = {
    // todo: number condition
    val target = y[Tensor[T]](1)
    if (target.nElement() == 1) {
      val storage = target.storage().array()
      val tmp1 = (input[Tensor[T]](1)(1) - input[Tensor[T]](2)(1)).storage().array()
      output = ev.max(ev.fromType(0),
        ev.plus(ev.times(tmp1(0), ev.negative(storage(0))), ev.fromType(margin)))
    } else {
      var _output = Tensor[T]()
      _output = input[Tensor[T]](1).clone
      _output.add(ev.fromType(-1), input[Tensor[T]](2)).mul(ev.fromType(-1)).cmul(target)
      _output.add(ev.fromType(margin))

      _output.cmax(0)
      output = _output.sum()

      if (sizeAverage) output = ev.divide(output, ev.fromType(target.size(1)))
    }
    output
  }

  override def updateGradInput(input: Table, y: Table): Table = {
    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T](1))
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T](1))
    // todo: number condition
    val target = y[Tensor[T]](1)
    if (target.nElement() == 1) {
      val storage = target.storage().array()
      val tmp1 = (input[Tensor[T]](1)(1) - input[Tensor[T]](2)(1)).storage().array()
      val dist = ev.toType[Double](tmp1(0)) * ev.toType[Double](ev.negative(storage(0))) + margin
      if (dist < 0) {
        gradInput[Tensor[T]](1).setValue(1, ev.fromType(0))
        gradInput[Tensor[T]](2).setValue(1, ev.fromType(0))
      } else {
        gradInput[Tensor[T]](1).setValue(1, ev.negative(storage(0)))
        gradInput[Tensor[T]](2).setValue(1, storage(0))
      }
    } else {
      var dist = Tensor[T]()
      dist = input[Tensor[T]](1).clone()
      dist.add(ev.fromType(-1), input[Tensor[T]](2))
      dist.mul(ev.fromType(-1)).cmul(target).add(ev.fromType(margin))

      val mask = Tensor[T]()
      mask.resizeAs(input[Tensor[T]](1)).copy(dist)

      mask.ge(dist, 0)
      gradInput[Tensor[T]](1).resizeAs(dist).copy(mask).mul(ev.fromType(-1)).cmul(target)
      gradInput[Tensor[T]](2).resizeAs(dist).copy(mask).cmul(target)

      if (sizeAverage) {
        gradInput[Tensor[T]](1).div(ev.fromType(target.size(1)))
        gradInput[Tensor[T]](2).div(ev.fromType(target.size(1)))
      }
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.MarginRankingCriterion($margin)"
  }
}
