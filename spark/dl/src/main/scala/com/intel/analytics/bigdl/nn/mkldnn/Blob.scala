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

/**
 * `Blob` contains two tensors, dense and native, which are a map of each other.
 * It's used in the layer which contains weights. For the weight, we should sync the
 * dense tensor to native tensor before `submit`. For the gradient, we should sync the native
 * tensor to dense tensor after `submit`.
 *
 * **We should call `resize` instead of calling the `resize` of `dense` and `native`.**
 *
 * The Blob has another attribute `_memoryData` and will not be determined when the blob created.
 * It can be determined when we initialize the primitives.
 *
 * @param _size the shape of Tensor, such as Array(4, 3, 224, 224)
 */
private[mkldnn] class Blob(_size: Array[Int]) extends Serializable {
  val dense: Tensor[Float] = Tensor[Float](_size)
  val native: DnnTensor[Float] = DnnTensor[Float](_size)

  private var _memoryData: MemoryData = _

  /**
   * it will copy the dense tensor to native tensor before `submit` reads the native tensor
   */
  def syncBeforeRead(): Unit = {
    native.copy(dense)
  }

  /**
   * it will copy the native tensor to dense tensor after `submit` updates the native tensor
   */
  def syncAfterWrite(): Unit = {
    dense.copy(native)
  }

  /**
   * MemoryData relevant of Native Tensor. The shape should be the same as `size` of Blob.
   * We can't only reserve the `layout` in MemoryData. Because for convolution,
   * we should reserve the whole MemoryData including `desc`, `desc primitive` and `primitive`.
   *
   * @param memoryData memory data you want.
   */
  def setMemoryData(memoryData: MemoryData): Unit = {
    require(size().deep == memoryData.shape.deep, s"You may assign wrong layout")
    _memoryData = memoryData
  }

  def memoryData(): MemoryData = {
    _memoryData
  }

  def zero(): Unit = {
    dense.zero()
    native.zero()
  }

  def copy(t: Tensor[Float]): Unit = {
    dense.copy(t)
    native.copy(t)
  }

  def size(): Array[Int] = {
    require(dense.size().deep == native.size().deep, s"The shape of Blob is not consistent")
    dense.size()
  }

  def size(index: Int): Int = {
    require(dense.size().deep == native.size().deep, s"The shape of Blob is not consistent")
    dense.size(index)
  }

  def resize(newSize: Array[Int]): this.type = {
    List(dense, native).foreach(_.resize(newSize))
    this
  }
}
