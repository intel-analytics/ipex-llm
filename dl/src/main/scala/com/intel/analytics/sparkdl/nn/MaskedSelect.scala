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


/**
 * Performs a torch.MaskedSelect on a Tensor. The mask is supplied as a tabular argument
 * with the input on the forward and backward passes.
 */

class MaskedSelect[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T]{

  @transient
  private var _maskIndices: Tensor[T] = null
  @transient
  private var _maskIndexBuffer: Tensor[T] = null
  @transient
  private var _gradBuffer: Tensor[T] = null
  @transient
  private var _gradMask: Tensor[T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    val _input = input[Tensor[T]](1)
    val mask = input[Tensor[T]](2)
    if (ev.toType[Double](mask.sum()) > 0) _input.maskedSelect(mask, output)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val _input = input[Tensor[T]](1)
    val mask = input[Tensor[T]](2)

    // ignore CudaTensor
    if (null == _maskIndices) _maskIndices = Tensor[T]()
    if (null == _maskIndexBuffer) _maskIndexBuffer = Tensor[T]()
    if (null == _gradBuffer) _gradBuffer = Tensor[T]()
    if (null == _gradMask) _gradMask = Tensor[T]()

    _maskIndexBuffer.range(1, mask.nElement())
    _maskIndexBuffer.resizeAs(mask)

    if (ev.toType[Double](mask.sum()) > 0) _maskIndexBuffer.maskedSelect(mask, _maskIndices)

    _gradBuffer.resize(_input.nElement()).zero()
    _gradBuffer.scatter(1, _maskIndices, gradOutput)
    _gradBuffer.resizeAs(_input)

    gradInput.insert(1, _gradBuffer)
    gradInput.insert(2, _gradMask.resizeAs(mask).zero())
    gradInput
  }

  override def toString(): String = {
    s"nn.MaskedSelect"
  }
}
