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
class MarginRankingCriterion[T: ClassTag]
(margin: Double = 1.0, sizeAverage: Boolean = true)
 (implicit ev: TensorNumeric[T]) extends Criterion[Table, T] {
  val gradInput = T()

  override def updateOutput(input: Table, y: Table): T = {
    // todo: number condition
    val target = y[Tensor[T]](1)
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)

    if (target.nElement() == 1) {
      val v1 = ev.minus(input1(Array(1)), input2(Array(1)))
      val v2 = ev.negative(target(Array(1)))
      output = ev.max(ev.fromType(0), ev.plus(ev.times(v1, v2), ev.fromType(margin)))
    } else {
      var _output = Tensor[T]()
      _output = input1.clone
      _output.add(ev.fromType(-1), input2).mul(ev.fromType(-1)).cmul(target)
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
    val input1 = input[Tensor[T]](1)
    val input2 = input[Tensor[T]](2)
    val gradInput1 = gradInput[Tensor[T]](1)
    val gradInput2 = gradInput[Tensor[T]](2)

    if (target.nElement() == 1) {
      val v1 = ev.minus(input1(Array(1)), input2(Array(1)))
      val v2 = target(Array(1))
      val dist = ev.toType[Double](v1) * ev.toType[Double](v2) * (-1) + margin
      if (dist < 0) {
        gradInput1.setValue(1, ev.fromType(0))
        gradInput2.setValue(1, ev.fromType(0))
      } else {
        gradInput1.setValue(1, ev.negative(v2))
        gradInput2.setValue(1, v2)
      }
    } else {
      val dist = input[Tensor[T]](1).clone()
      dist.add(ev.fromType(-1), input[Tensor[T]](2))
      dist.mul(ev.fromType(-1)).cmul(target).add(ev.fromType(margin))

      val mask = dist.clone()

      mask.ge(dist, 0)
      gradInput1.resizeAs(dist).copy(mask).mul(ev.fromType(-1)).cmul(target)
      gradInput2.resizeAs(dist).copy(mask).cmul(target)

      if (sizeAverage) {
        gradInput1.div(ev.fromType(target.size(1)))
        gradInput2.div(ev.fromType(target.size(1)))
      }
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.MarginRankingCriterion($margin)"
  }
}
