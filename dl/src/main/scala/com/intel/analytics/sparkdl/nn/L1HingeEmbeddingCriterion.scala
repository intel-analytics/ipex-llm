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
import scala.util.Random

/**
 * Creates a criterion that measures the loss given an input x = {x1, x2},
  * a table of two Tensors, and a label y (1 or -1):
 * @param margin
 */
class L1HingeEmbeddingCriterion[T: ClassTag](margin: Double = 1)
 (implicit ev: TensorNumeric[T]) extends Criterion[Table, T]{
  val gradInput = T()

  private def mathSign(t: T): T = {
    var res = 0
    if (ev.isGreater(t, ev.fromType(0))) {
      res = 1
    } else if (ev.isGreater(ev.fromType(0), t)) {
      res = -1
    } else {
      res = 2 * (Random.nextInt(2) + 1) - 3
    }
    ev.fromType(res)
  }

  override def updateOutput(input: Table, target: Table): T = {
    require(target[Tensor[T]](1).isContiguous())
    val _output = (input[Tensor[T]](1)-input[Tensor[T]](2)).abs().pow(ev.fromType(1))
    output = ev.pow(_output.sum(), ev.fromType(1))
    val targetArr = target[Tensor[T]](1).storage().array()
    if (targetArr(0) == -1) output = ev.max(ev.fromType(0), ev.minus(ev.fromType(margin), output))
    output
  }

  override def updateGradInput(input: Table, target: Table): Table = {
    require(target[Tensor[T]](1).isContiguous())
    val targetArr = target[Tensor[T]](1).storage().array()

    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T])
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T])

    val gradInput1 = gradInput[Tensor[T]](1)
    val gradInput2 = gradInput[Tensor[T]](2)

    gradInput1.resizeAs(input[Tensor[T]](1)).copy(input[Tensor[T]](1)).
      add(ev.fromType(-1), input[Tensor[T]](2))
    gradInput2.resizeAs(input[Tensor[T]](2))

    val dist = gradInput1.abs().sum()
    gradInput1.apply1(mathSign)

    if (targetArr(0) == -1) {
      if (ev.isGreater(dist, ev.fromType(margin))) {
        gradInput1.zero()
      } else {
        gradInput1.mul(ev.fromType(-1))
      }
    }
    gradInput2.zero().add(ev.fromType(-1), gradInput1)
    gradInput
  }

  override def toString(): String = {
    s"nn.L1HingeEmbeddingCriterion ($margin)"
  }
}
