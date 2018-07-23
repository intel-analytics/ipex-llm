/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

private[mkldnn] class Blob(private val _size: Array[Int]) extends Serializable {
  val dense: Tensor[Float] = Tensor[Float](_size)
  val native: DnnTensor[Float] = DnnTensor[Float](_size)

  private var _shape: MemoryData = _

  def sync(isDense2Native: Boolean = true): Unit = {
    if (isDense2Native) {
      native.copy(dense)
    } else {
      dense.copy(native)
    }
  }

  def setShape(shape: MemoryData): Unit = {
    _shape = shape
  }

  def shape(): MemoryData = _shape

  def zero(): Unit = {
    dense.zero()
    native.zero()
  }

  def copy(t: Tensor[Float]): Unit = {
    dense.copy(t)
    native.copy(t)
  }

  def size(): Array[Int] = _size

  def size(index: Int): Int = dense.size(index)
}
