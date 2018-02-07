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

import com.intel.analytics.bigdl.mkl.{Memory, MklDnn}
import com.intel.analytics.bigdl.mkl.MklDnn.EngineType
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, Tensor}

object MklDnnOps {

  private val engineType = EngineType.cpu

  def engineCreate(index : Int) : Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EngineCreate(engineType, index)
  }

  def engineDestroy(engine: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EngineDestroy(engine)
  }

  def streamCreate(streamKind: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamCreate(streamKind)
  }

  def streamSubmit(loc: Long, block: Int, primitives: Array[Long], length: Int): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamSubmit(loc, block, primitives)
  }

  def streamWait(loc: Long, block: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamWait(loc, block)
  }

  def streamDestroy(loc: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.StreamDestroy(loc)
  }

  def memoryDescInit(ndims: Int, dims: Array[Int], dataType: Int, dataFormat: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryDescInit(ndims, dims, dataType, dataFormat)
  }

  def memoryPrimitiveDescCreate(desc: Long, engine: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryPrimitiveDescCreate(desc, engine)
  }

  def memoryGetDataHandle(memory: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryGetDataHandle(memory)
  }

  def memorySetDataHandle(memory: Long, data: Tensor[Float], offset: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemorySetDataHandle(memory, data.storage().array(), offset)
  }

  def memoryReleaseDataHandle(data: Tensor[Float], ptr: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryReleaseDataHandle(data.storage().array(), ptr)
  }

  def primitiveCreate0(desc: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveCreate0(desc)
  }

  def primitiveCreate2(desc: Long, inputs: Array[Long], indexes: Array[Int], length1: Int,
                       outputs: Array[Long], lenght2 : Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveCreate2(desc, inputs, indexes, length1, outputs, lenght2)
  }

  def primitiveDescCreate(opDesc: Long, engine: Long, hingForwardPrimitiveDesc: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDescCreate(opDesc, engine, hingForwardPrimitiveDesc)
  }

  def primitiveDescDestroy(desc: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDescDestroy(desc)
  }

  def primitiveDestroy(primitive: Long): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDestroy(primitive)
  }

  def eltwiseForwardDescInit(propKind: Int, algKind: Int, srcDesc: Long,
                             alpha: Float, beta: Float): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EltwiseForwardDescInit(propKind, algKind, srcDesc,
      alpha, beta)
  }

  def eltwiseBackwardDescInit(algKind: Int, diffDataDesc: Long, dataDesc: Long,
                              alpha: Float, beta: Float): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.EltwiseBackwardDescInit(algKind, diffDataDesc, dataDesc,
      alpha, beta)
  }

  def convForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, weights_desc: Long,
    bias_desc: Long, dst_desc: Long, strides: Array[Int],
    padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
      MklDnn.ConvForwardDescInit(prop_kind, alg_kind, src_desc, weights_desc,
        bias_desc, dst_desc, strides, padding_l, padding_r, padding_kind)
  }

  def convBackwardWeightsDescInit(alg_kind: Int, src_desc: Long, diff_weights_desc: Long,
    diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int],
    padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.ConvBackwardWeightsDescInit(alg_kind, src_desc, diff_weights_desc,
      diff_bias_desc, diff_dst_desc, strides, padding_l, padding_r, padding_kind)
  }

  def convBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
    diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.ConvBackwardDataDescInit(alg_kind, diff_src_desc, weights_desc,
      diff_dst_desc, strides, padding_l, padding_r, padding_kind)
  }

  def poolingForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, dst_desc: Long,
                             strides: Array[Int], kernel: Array[Int], padding_l: Array[Int],
                             padding_r: Array[Int], padding_kind: Int): Long = {
    MklDnn.PoolingForwardDescInit(prop_kind, alg_kind, src_desc, dst_desc, strides, kernel,
      padding_l, padding_r, padding_kind)
  }

  def poolingBackwardDescInit(alg_kind: Int, diff_src_desc: Long, diff_dst_desc: Long,
                              strides: Array[Int], kernel: Array[Int], padding_l: Array[Int],
                              padding_r: Array[Int], padding_kind: Int): Long = {
    MklDnn.PoolingBackwardDescInit(alg_kind, diff_src_desc, diff_dst_desc, strides, kernel,
      padding_l, padding_r, padding_kind)
  }

  def reorderPrimitiveDescCreate(input: Long, output: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.ReorderPrimitiveDescCreate(input, output)
  }

  def memoryPrimitiveDescEqual(lhs: Long, rhs: Long): Int = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.MemoryPrimitiveDescEqual(lhs, rhs)
  }

  def primitiveGetPrimitiveDesc(primitive: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveGetPrimitiveDesc(primitive)
  }

  def primitiveDescQueryPd(primitive: Long, what: Int, index: Int): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDescQueryPd(primitive, what, index)
  }

  def primitiveDescQueryMemory(primitive_desc: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.PrimitiveDescQueryMemory(primitive_desc)
  }

  /**
   * @param user_memory
   * @param prim_memory_pd not a pointer, but direct const_mkldnn_primitive_desc_t
   * @param user_to_prim
   * @return
   */
  def prepareReorder(user_memory: Long, prim_memory_pd: Long,
                     user_to_prim: Boolean): (Long, Long) = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    val user_memory_pd = MklDnn.PrimitiveGetPrimitiveDesc(user_memory)
    var reorder: Long = 0L
    var prim_memory: Long = 0L
    if (MklDnn.MemoryPrimitiveDescEqual(user_memory_pd, prim_memory_pd) == 0) {
      prim_memory = MklDnn.PrimitiveCreate0(prim_memory_pd)
      var reorder_pd: Long = 0L
      if (user_to_prim) {
        reorder_pd = MklDnn.ReorderPrimitiveDescCreate(user_memory_pd, prim_memory_pd)
        val inputs = Array(user_memory)
        val indexes = Array(0)
        val outputs = Array(prim_memory)
        reorder = MklDnnOps.primitiveCreate2(reorder_pd, inputs, indexes, 1, outputs, 1)
      } else {
        reorder_pd = MklDnn.ReorderPrimitiveDescCreate(prim_memory_pd, user_memory_pd)
        val inputs = Array(prim_memory)
        val indexes = Array(0)
        val outputs = Array(user_memory)
        reorder = MklDnnOps.primitiveCreate2(reorder_pd, inputs, indexes, 1, outputs, 1)
      }
      MklDnn.PrimitiveDescDestroy(reorder_pd)
    }
    return (reorder, prim_memory)
  }

  def initDataMemory(dim: Int, dims: Array[Int], memoryFormat: Int,
                     dataType: Int, engine: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    val prim_md = MklDnn.MemoryDescInit(dim, dims, dataType, memoryFormat)
    val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
    val memory = MklDnn.PrimitiveCreate0(user_pd)
    primitiveDescDestroy(user_pd)
    memory
  }

  def createMemoryPrimitive(prim_md: Long, engine: Long): Long = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
    val memory = MklDnn.PrimitiveCreate0(user_pd)
    primitiveDescDestroy(user_pd)
    memory
  }


  def streamSubmit(loc: Long, block: Int, primitives: Array[Long], length: Int,
                   memory_primitives: Array[Long], buffers: Array[Tensor[Float]]): Unit = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    require(memory_primitives.length == buffers.length)

    val handle = new Array[Long](memory_primitives.length)
    for (i <- 0 to memory_primitives.length - 1) {
      if (memory_primitives(i) != 0L) {
        if (buffers(i).isInstanceOf[MklDnnTensor[Float]]) {
          Memory.SetDataHandle(memory_primitives(i),
            buffers(i).asInstanceOf[MklDnnTensor[Float]].ptr, 0)
        } else {
          handle(i) = MklDnnOps.memorySetDataHandle(
            memory_primitives(i), buffers(i), buffers(i).storageOffset() - 1)
        }
      }
    }

    MklDnn.StreamSubmit(loc, block, primitives)

    for (i <- 0 to memory_primitives.length - 1) {
      if (memory_primitives(i) != 0L) {
         MklDnnOps.memoryReleaseDataHandle(buffers(i), handle(i))
      }
    }
  }

  def getFormat(memoryDesc: Long): Int = {
    require(MklDnn.isLoaded, "mkldnn isn't loaded")
    MklDnn.getFormat(memoryDesc)
  }

}
