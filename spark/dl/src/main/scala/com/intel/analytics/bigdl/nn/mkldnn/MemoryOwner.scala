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

import scala.collection.mutable.ArrayBuffer

trait MemoryOwner {
  @transient
  private lazy val _nativeMemory: ArrayBuffer[MklDnnNativeMemory] =
    new ArrayBuffer[MklDnnNativeMemory]()

  @transient
  private lazy val _tensorMMaps: ArrayBuffer[TensorMMap] =
    new ArrayBuffer[TensorMMap]()

  @transient
  private  var _reorderManager: ReorderManager = _

  def registerMklNativeMemory(m: MklDnnNativeMemory): Unit = {
    _nativeMemory.append(m)
  }

  def registerTensorMMap(m: TensorMMap): Unit = {
    _tensorMMaps.append(m)
  }

  def registerReorderManager(m: ReorderManager): Unit = {
    require(_reorderManager == null, "reorderManager should be null in MemoryOwner")
    _reorderManager = m
  }

  def releaseNativeMklDnnMemory(): Unit = {
    _nativeMemory.foreach(m => {
      if (!m.isUndefOrError) {
        m.release()
        m.reset()
      }
    })
    _nativeMemory.clear()
    _tensorMMaps.foreach(_.release())
    _tensorMMaps.clear()
    if (_reorderManager != null) {
      _reorderManager.release()
    }
  }
}
