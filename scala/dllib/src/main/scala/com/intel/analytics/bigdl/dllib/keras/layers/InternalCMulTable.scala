/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.CMulTable
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

private[zoo] class InternalCMulTable[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends CMulTable[T] {
  private var expandLayer: AbstractModule[Tensor[T], Tensor[T], T] = null

  override def updateOutput(input: Table): Tensor[T] = {
    var i = 1
    var sizes: Array[Int] = null
    while (i <= input.length() && sizes == null) {
      if (input[Tensor[T]](i).size(1) != 1) {
        sizes = input[Tensor[T]](i).size()
        expandLayer = InternalExpand(sizes)
      }
      i += 1
    }
    val newInput = if (sizes != null) {
      val _expandInput = T()
      i = 1
      while (i <= input.length()) {
        if (input[Tensor[T]](i).size(1) == 1) {
          _expandInput(i) = expandLayer.forward(input[Tensor[T]](i))
        } else _expandInput(i) = input(i)
        i += 1
      }
      _expandInput
    } else input
    output = super.updateOutput(newInput)
    return output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 1
    var sizes: Array[Int] = null
    while (i <= input.length() && sizes == null) {
      if (input[Tensor[T]](i).size(1) != 1) {
        sizes = input[Tensor[T]](i).size()
        expandLayer = InternalExpand(sizes)
      }
      i += 1
    }
    val newInput = if (sizes != null) {
      val _expandInput = T()
      i = 1
      while (i <= input.length()) {
        if (input[Tensor[T]](i).size(1) == 1) {
          _expandInput(i) = expandLayer.forward(input[Tensor[T]](i))
        } else _expandInput(i) = input(i)
        i += 1
      }
      _expandInput
    } else input

    gradInput = super.updateGradInput(newInput, gradOutput)
    i = 1
    if (sizes != null) {
      while (i <= input.length()) {
        if (input[Tensor[T]](i).size(1) == 1) {
          gradInput(i) = expandLayer.backward(input[Tensor[T]](i), gradInput[Tensor[T]](i))
        }
        i += 1
      }
    }
    gradInput
  }

  override def toString: String = s"InternalCMulTable()"

  override def canEqual(other: Any): Boolean = other.isInstanceOf[InternalCMulTable[T]]
}

private[zoo] object InternalCMulTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : InternalCMulTable[T] = {
    new InternalCMulTable[T]()
  }
}
