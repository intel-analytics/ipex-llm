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
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag

class CosineDistance[T: ClassTag]()(
  implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {

  @transient
  var buffer: Tensor[T] = null
  @transient
  var w1: Tensor[T] = null
  @transient
  var w22: Tensor[T] = null
  @transient
  var w: Tensor[T] = null
  @transient
  var w32: Tensor[T] = null
  @transient
  var ones: Tensor[T] = null

  def makeContiguous(input1: Tensor[T], input2: Tensor[T]): (Tensor[T], Tensor[T]) = {
    var _input1 = Tensor[T]()
    var _input2 = Tensor[T]()

    if (!input1.isContiguous()) {
      _input1.resizeAs(input1).copy(input1)
    } else {
      _input1 = input1
    }
    if (!input2.isContiguous()) {
      _input2.resizeAs(input2).copy(input2)
    } else {
      _input2 = input2
    }
    (_input1, _input2)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    if (null == buffer) buffer = Tensor[T]()
    if (null == w1) w1 = Tensor[T]()
    if (null == w22) w22 = Tensor[T]()
    if (null == w) w = Tensor[T]()
    if (null == w32) w32 = Tensor[T]()
    if (null == ones) ones = Tensor[T]()

    var (input1, input2) = makeContiguous(input[Tensor[T]](1), input[Tensor[T]](2))
    if (input1.dim() == 1) {
      input1 = input1.view(1, input1.nElement())
      input2 = input2.view(1, input2.nElement())
    }

    buffer.resizeAs(input1).cmul(input1, input2)
    w1.sum(buffer, 2)

    val epsilon = 1e-12
    buffer.cmul(input1, input1)
    w22.sum(buffer, 2).add(ev.fromType(epsilon))
    ones.resizeAs(w22).fill(ev.fromType(1))
    w22.cdiv(ones, w22)
    w.resizeAs(w22).copy(w22)

    buffer.cmul(input2, input2)
    w32.sum(buffer, 2).add(ev.fromType(epsilon))
    w32.cdiv(ones, w32)
    w.cmul(w32)
    w.sqrt()

    output.resizeAs(w1).cmul(w1, w)
    output.resize(input1.size(1))

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    var no_batch = false
    var (v1, v2) = makeContiguous(input[Tensor[T]](1), input[Tensor[T]](2))

    if (v1.dim() == 1) {
      v1 = v1.view(1, v1.nElement())
      v2 = v2.view(1, v2.nElement())
      no_batch = true
    }

    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T])
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T])

    val gw1 = gradInput[Tensor[T]](1)
    val gw2 = gradInput[Tensor[T]](2)

    gw1.resizeAs(v1).copy(v2)
    gw2.resizeAs(v1).copy(v1)

    buffer.resizeAs(w1).cmul(w1, w22)
    gw1.addcmul(ev.fromType(-1), buffer.expandAs(v1), v1)
    gw1.cmul(w.expandAs(v1))

    buffer.resizeAs(w1).cmul(w1, w32)
    gw2.addcmul(ev.fromType(-1), buffer.expandAs(v1), v2)
    gw2.cmul(w.expandAs(v1))

    val go = gradOutput.view(gradOutput.nElement(), 1).expandAs(v1)
    gw1.cmul(go)
    gw2.cmul(go)

    if (no_batch) {
      gradInput[Tensor[T]](1).resize(gw1.size(2))
      gradInput[Tensor[T]](2).resize(gw2.size(2))
    }
    gradInput
  }

  override def toString(): String = {
    s"nn.CosineDistance"
  }
}
