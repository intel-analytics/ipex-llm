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

import com.intel.analytics.bigdl.mkl.{DataType, Memory}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.{DnnTensor, FloatType, Tensor}

import scala.reflect.ClassTag

/**
 * `TensorMMap` contains two tensors, dense and native, which are a map of each other.
 * It's used in the layer which contains weights. For the weight, we should sync the
 * dense tensor to native tensor before `submit`. For the gradient, we should sync the native
 * tensor to dense tensor after `submit`.
 *
 * @param _size the shape of Tensor, such as Array(4, 3, 224, 224)
 */
private[bigdl] class TensorMMap(_size: Array[Int])(implicit owner: MemoryOwner)
  extends Serializable {
  // dense weight on heap is used to optimizer and so on, which is exposed to
  // AbstractModule level.
  val dense: Tensor[Float] = Tensor[Float](_size)

  def native[T]: DnnTensor[T] = {
    _native.asInstanceOf[DnnTensor[T]]
  }

  // the native DnnTensor. It's allocate at runtime when do primitive initialization.
  // it has two benefits, the first is the clone will only clone one copy of weights and gradients
  // before primitive initialized and the second is that we can determined the size, type, format
  // when do initializing primitive.
  @transient private var _native: DnnTensor[_] = _

  @transient private var _from: MemoryData = null
  @transient private var _to: MemoryData = null
  @transient private var _reorder: ReorderMemory = null

  @transient private var _heapData: HeapData = null

  def heapData: HeapData = _heapData

  def sync(): Unit = {
    require(_reorder != null && _native != null,
      "you should initialize the native relevant resources first")
    _from match {
      case _: HeapData => _reorder.forward(this.dense)
      case _: NativeData => _reorder.forward(this.native)
    }
  }

  /**
   * set the dense <-> native map, maintain the format to reorder
   *
   * Note, it will only create the native tensor based on the size and will not
   * do the reorder. So you should call `sync()` by manual.
   *
   * @param from the source tensor memory data, could be HeapData or NativeData
   * @param to the dest tensor memory data, could be HeapData or NativeData
   * @param runtime the mkldnn runtime for reorder operation
   */
  def setMemoryData(from: MemoryData, to: MemoryData, runtime: MklDnnRuntime): Unit = {
    require(_from == null && _to == null, "you only can set once the memory data")
    _from = from
    _to = to

    _reorder = ReorderMemory(to)
    _reorder.setRuntime(runtime)
    _reorder.initFwdPrimitives(Array(_from), InferencePhase)

    _from match {
      case _: HeapData =>
        this._native = _reorder.output.asInstanceOf[DnnTensor[Float]]
        _heapData = _from.asInstanceOf[HeapData]
      case _: NativeData =>
        // the native tensor size should be determined by the memory description
        // other wise will be segment fault
        this._native = DnnTensor[Float](Memory.GetPaddingShape(_from.getMemoryDescription()))
        // the native initialize value should be all zeros.
        this._native.zero()
        _reorder.output.toTensor[Float].set(this.dense)
        _heapData = _to.asInstanceOf[HeapData]
    }
  }

  def zero(): Unit = {
    dense.zero()
    if (native != null) {
      native.zero()
    }
  }

  def copy(t: Tensor[Float]): Unit = {
    dense.copy(t)
  }

  def size(): Array[Int] = {
    dense.size()
  }

  def size(index: Int): Int = {
    dense.size(index)
  }

  def release(): Unit = {
    if (native != null) {
      native.release()
    }
  }

  def setNative(another: TensorMMap): Unit = {
    if (native != null && another.native != null) {
      native.set(another.native.asInstanceOf[Tensor[_]])
    }
  }
}
