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

import com.intel.analytics.bigdl.dnnl.{DNNL, Memory, Engine => DnnEngine, Stream => DnnStream}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

import scala.collection.mutable

private[mkldnn] object MklDnnOps {

  def memorySetDataHandle(memory: Long, data: Tensor[Float], offset: Int): Long = {
    require(DNNL.isLoaded, "mkldnn isn't loaded")
    val ret = DNNL.MemorySetDataHandle(memory, data.storage().array(), offset)
    ret
  }

  def memoryReleaseDataHandle(data: Tensor[Float], ptr: Long): Unit = {
    require(DNNL.isLoaded, "mkldnn isn't loaded")
    DNNL.MemoryReleaseDataHandle(data.storage().array(), ptr)
  }

  def streamSubmit(primitives: Array[Long], stream: Long,
                   execArgs: mutable.Map[Int, Long],
                   execTensors: mutable.Map[Int, Tensor[Float@unchecked]]): Unit = {
    // the tensor maybe Tensor[Byte]. so use the unchecked to handle this
    require(DNNL.isLoaded, "mkldnn isn't loaded")
    require(execArgs.size == execTensors.size, execArgs.size + " " + execTensors.size)

    val indexes = new Array[Int](execArgs.size)
    val memories = new Array[Long](execArgs.size)
    val handle = new Array[Long](execArgs.size)

    // TODO: would like to set data h
    execArgs.zipWithIndex.foreach {
      case ((argIdx, argMemory), i) =>
        if (execTensors.get(argIdx).get.isInstanceOf[DnnTensor[_]]) {
          Memory.SetDataHandle(argMemory,
            execTensors.get(argIdx).get.asInstanceOf[DnnTensor[Float]].storageAddress(),
            0)
        } else {
          handle(i) = MklDnnOps.memorySetDataHandle(
            argMemory, execTensors.get(argIdx).get,
            execTensors.get(argIdx).get.storageOffset() - 1)
        }
        indexes(i) = argIdx
        memories(i) = argMemory
    }
    DnnStream.Submit(primitives(0), stream, execArgs.size, indexes, memories)

    handle.indices.foreach(i => {
      if (handle(i) != 0L) {
        MklDnnOps.memoryReleaseDataHandle(execTensors.get(indexes(i)).get, handle(i))
      }
    })

  }
}
