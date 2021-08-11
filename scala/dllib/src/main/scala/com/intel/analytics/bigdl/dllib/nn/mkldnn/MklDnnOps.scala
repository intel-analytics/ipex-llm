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

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn, Engine => DnnEngine, Stream => DnnStream}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

private[mkldnn] object MklDnnOps {
  def memorySetDataHandle(memory: Long, data: Tensor[Float], offset: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemorySetDataHandle(memory, data.storage().array(), offset)
  }

  def memoryReleaseDataHandle(data: Tensor[Float], ptr: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryReleaseDataHandle(data.storage().array(), ptr)
  }

  def streamSubmit(loc: Long, block: Int, primitives: Array[Long], length: Int,
    memory_primitives: Array[Long],
    buffers: Array[Tensor[Float@unchecked]]): Unit = {
    // the tensor maybe Tensor[Byte]. so use the unchecked to handle this
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    require(memory_primitives.length == buffers.length)

    val handle = new Array[Long](memory_primitives.length)
    for (i <- memory_primitives.indices) {
      if (memory_primitives(i) != 0L) {
        if (buffers(i).isInstanceOf[DnnTensor[_]]) {
          Memory.SetDataHandle(memory_primitives(i),
            buffers(i).asInstanceOf[DnnTensor[Float]].storageAddress(), 0)
        } else {
          handle(i) = MklDnnOps.memorySetDataHandle(
            memory_primitives(i), buffers(i), buffers(i).storageOffset() - 1)
        }
      }
    }

    DnnStream.Submit(loc, block, primitives)

    for (i <- memory_primitives.indices) {
      if (handle(i) != 0L) {
         MklDnnOps.memoryReleaseDataHandle(buffers(i), handle(i))
      }
    }
  }
}
