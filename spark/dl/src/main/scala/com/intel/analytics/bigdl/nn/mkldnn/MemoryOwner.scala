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

import com.intel.analytics.bigdl.tensor.DnnTensor
import scala.collection.mutable.ArrayBuffer

/**
 * This trait is a owner of MKLDNN native memory. It will track all native resources
 * (Primitives, tensor, ReorderMemory). You can call releaseNativeMklDnnMemory to relase all the
 * memory at once. These resources will require an implicit MemoryOwner at
 * the constructors. The constructors of the resources will register themselves to the MemoryOwner.
 * For DNN Layer classes, they extends MemoryOwner and have a implicit value of "this" as a
 * MemoryOwner. ReorderMemory is a kind of special resource. They can be a normal layer or a
 * resource of another layer.
 */
private[bigdl] trait MemoryOwner {
  @transient
  private lazy val _nativeMemory: ArrayBuffer[MklDnnNativeMemory] =
    new ArrayBuffer[MklDnnNativeMemory]()

  @transient
  private lazy val _tensors: ArrayBuffer[DnnTensor[_]] =
    new ArrayBuffer[DnnTensor[_]]()

  @transient
  private lazy val _reorderMemory: ArrayBuffer[ReorderMemory] = new ArrayBuffer[ReorderMemory]()

  def registerMklNativeMemory(m: MklDnnNativeMemory): Unit = {
    _nativeMemory.append(m)
  }

  def registerTensor(m: DnnTensor[_]): Unit = {
    _tensors.append(m)
  }

  def registerReorderMemory(m: ReorderMemory): Unit = {
    _reorderMemory.append(m)
  }

  def releaseNativeMklDnnMemory(): Unit = {
    _nativeMemory.foreach(m => {
      if (!m.isUndefOrError) {
        m.release()
        m.reset()
      }
    })
    _nativeMemory.clear()
    _tensors.foreach(_.release())
    _tensors.clear()
    _reorderMemory.foreach(_.release())
    _reorderMemory.clear()
  }
}
