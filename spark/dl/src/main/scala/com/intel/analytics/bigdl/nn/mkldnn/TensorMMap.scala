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

/**
 * `TensorMMap` contains two tensors, dense and native, which are a map of each other.
 * It's used in the layer which contains weights. For the weight, we should sync the
 * dense tensor to native tensor before `submit`. For the gradient, we should sync the native
 * tensor to dense tensor after `submit`.
 *
 * The `setMemoryData` requires the elements number should be consistent. If the shape is not,
 * it will reshape first.
 *
 * The TensorMMap has another attribute `_memoryData` and will not be determined when the blob created
 * It can be determined when we initialize the primitives.
 *
 * @param _size the shape of Tensor, such as Array(4, 3, 224, 224)
 */
private[mkldnn] class TensorMMap(_size: Array[Int]) extends Serializable {
  // dense weight on heap is used to optimizer and so on, which is exposed to
  // AbstractModule level.
  val dense: Tensor[Float] = Tensor[Float](_size)

  def native[T]: DnnTensor[T] = {
    _native.asInstanceOf[DnnTensor[T]]
  }

  @transient private var _native: DnnTensor[_] = _
  @transient private var _heapData: HeapData = null
  @transient private var _nativeData: NativeData = null
  @transient private var _reorderForward: ReorderMemory = null
  @transient private var _reorderBackward: ReorderMemory = null

  def heapData: HeapData = _heapData

  /**
   * it will copy the dense tensor to native tensor before `submit` reads the native tensor
   */
  def syncToNative(): Unit = {
    _reorderForward.forward(this.dense).asInstanceOf[DnnTensor[Float]]
  }

  /**
   * it will copy the native tensor to dense tensor after `submit` updates the native tensor
   */
  def syncToHeap(): Unit = {
    _reorderBackward.forward(this.native).asInstanceOf[Tensor[Float]]
  }

  def setMemoryData(dense: HeapData, native: NativeData, runtime: MklDnnRuntime): Unit = {
    _heapData = dense
    _nativeData = native

    _reorderForward = ReorderMemory(_nativeData)
    _reorderForward.setRuntime(runtime)
    _reorderForward.initFwdPrimitives(Array(_heapData), InferencePhase)

    this._native = _reorderForward.updateOutput(this.dense).asInstanceOf[DnnTensor[Float]]

    _reorderBackward = ReorderMemory(_heapData)
    _reorderBackward.setRuntime(runtime)
    _reorderBackward.initFwdPrimitives(Array(_nativeData), InferencePhase)
    _reorderBackward.output.toTensor[Float].set(this.dense)
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
}
