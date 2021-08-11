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

import com.intel.analytics.bigdl.mkl.MklDnn

abstract class MklDnnNativeMemory(protected var __ptr: Long)(implicit owner: MemoryOwner)
extends Releasable {
  private val UNDEFINED: Long = -1
  private val ERROR: Long = 0

  owner.registerResource(this)

  def isUndefOrError : Boolean = __ptr == UNDEFINED || __ptr == ERROR
  def release(): Unit = {
    if (!isUndefOrError) {
      doRelease()
      reset()
    }
  }

  def doRelease(): Unit
  def ptr: Long = __ptr
  def reset(): Unit = {
    __ptr = ERROR
  }

}
class MklMemoryPrimitiveDesc(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = MklDnn.PrimitiveDescDestroy(ptr)
}

class MklMemoryAttr(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = MklDnn.DestroyAttr(ptr)
}

class MklMemoryPostOps(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = MklDnn.DestroyPostOps(ptr)
}

// All *DescInit memory objects share the same dealloactor
class MklMemoryDescInit(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = MklDnn.FreeMemoryDescInit(ptr)
}

class MklMemoryPrimitive(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = MklDnn.PrimitiveDestroy(ptr)
}


object MklDnnMemory {

  // scalastyle:off
  def MemoryDescInit(ndims: Int, dims: Array[Int], dataType: Int, dataFormat: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.MemoryDescInit(ndims, dims, dataType, dataFormat)).ptr
  }

  def EltwiseForwardDescInit(propKind: Int, algKind: Int, srcDesc: Long, alpha: Float,
    beta: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.EltwiseForwardDescInit(propKind, algKind, srcDesc, alpha, beta)).ptr
  }

  def EltwiseBackwardDescInit(algKind: Int, diffDataDesc: Long, dataDesc: Long, alpha: Float,
    beta: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.EltwiseBackwardDescInit(algKind, diffDataDesc, dataDesc, alpha, beta)).ptr
  }

  def LinearForwardDescInit(propKind: Int, srcMemDesc: Long, weightMemDesc: Long,
    biasMemDesc: Long, dstMemDesc: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.LinearForwardDescInit(propKind, srcMemDesc, weightMemDesc, biasMemDesc, dstMemDesc)).ptr
  }

  def LinearBackwardDataDescInit(diffSrcMemDesc: Long, weightMemDesc: Long, diffDstMemDesc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc, diffDstMemDesc)).ptr
  }

  def LinearBackwardWeightsDescInit(srcMemDesc: Long, diffWeightMemDesc: Long,
    diffBiasMemDesc: Long, diffDstMemDesc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.LinearBackwardWeightsDescInit(srcMemDesc,
        diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc)).ptr
  }

  def BatchNormForwardDescInit(propKind: Int, srcMemDesc: Long, epsilon: Float, flags: Long)
  (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.BatchNormForwardDescInit(propKind, srcMemDesc, epsilon, flags)).ptr
  }

  def BatchNormBackwardDescInit(prop_kind: Int, diffDstMemDesc: Long, srcMemDesc: Long,
    epsilon: Float, flags: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.BatchNormBackwardDescInit(prop_kind, diffDstMemDesc, srcMemDesc, epsilon, flags)).ptr
  }

  def SoftMaxForwardDescInit(prop_kind: Int, dataDesc: Long, axis: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.SoftMaxForwardDescInit(prop_kind, dataDesc, axis)).ptr
  }

  def SoftMaxBackwardDescInit(propKind: Int, diffDesc: Long, dstDesc: Long,
    axis: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(MklDnn.SoftMaxBackwardDescInit(diffDesc, dstDesc, axis)).ptr
  }

  def ConvForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, weights_desc: Long,
    bias_desc: Long, dst_desc: Long, strides: Array[Int], padding_l: Array[Int],
    padding_r: Array[Int], padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.ConvForwardDescInit(prop_kind, alg_kind, src_desc, weights_desc,
        bias_desc, dst_desc, strides, padding_l,
        padding_r, padding_kind)).ptr
  }

  def DilatedConvForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long,
    weights_desc: Long, bias_desc: Long, dst_desc: Long, strides: Array[Int],
    dilates: Array[Int], padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.DilatedConvForwardDescInit(prop_kind, alg_kind, src_desc,
        weights_desc, bias_desc, dst_desc, strides,
        dilates, padding_l, padding_r, padding_kind)).ptr
  }

  def ConvBackwardWeightsDescInit(alg_kind: Int, src_desc: Long, diff_weights_desc: Long,
    diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int],
    padding_r: Array[Int], padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.ConvBackwardWeightsDescInit(alg_kind, src_desc, diff_weights_desc,
        diff_bias_desc, diff_dst_desc, strides, padding_l,
        padding_r, padding_kind)).ptr
  }

  def DilatedConvBackwardWeightsDescInit(alg_kind: Int, src_desc: Long, diff_weights_desc: Long,
    diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int], dilates: Array[Int],
    padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.DilatedConvBackwardWeightsDescInit(alg_kind: Int, src_desc: Long,
        diff_weights_desc: Long,
        diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int], dilates: Array[Int],
        padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int)).ptr
  }

  def ConvBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
    diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.ConvBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
        diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
        padding_kind: Int)).ptr
  }

  def DilatedConvBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
    diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], dilates: Array[Int],
    padding_r: Array[Int], padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.DilatedConvBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
        diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], dilates: Array[Int],
        padding_r: Array[Int], padding_kind: Int)).ptr
  }

  def PoolingForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, dst_desc: Long,
    strides: Array[Int], kernel: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.PoolingForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, dst_desc: Long,
        strides: Array[Int], kernel: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
        padding_kind: Int)).ptr
  }

  def PoolingBackwardDescInit(alg_kind: Int, diff_src_desc: Long, diff_dst_desc: Long,
    strides: Array[Int], kernel: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.PoolingBackwardDescInit(alg_kind: Int, diff_src_desc: Long, diff_dst_desc: Long,
        strides: Array[Int], kernel: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
        padding_kind: Int)).ptr
  }

  def LRNForwardDescInit(prop_kind: Int, alg_kind: Int, data_desc: Long, local_size: Int,
    alpha: Float, beta: Float, k: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.LRNForwardDescInit(prop_kind: Int, alg_kind: Int, data_desc: Long, local_size: Int,
        alpha: Float, beta: Float, k: Float)).ptr
  }

  def LRNBackwardDescInit(alg_kind: Int, diff_data_desc: Long, data_desc: Long, local_size: Int,
    alpha: Float, beta: Float, k: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.LRNBackwardDescInit(alg_kind: Int, diff_data_desc: Long, data_desc: Long,
        local_size: Int, alpha: Float, beta: Float, k: Float)).ptr
  }

  def RNNCellDescInit(kind: Int, f: Int, flags: Int, alpha: Float, clipping: Float)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.RNNCellDescInit(kind: Int, f: Int, flags: Int, alpha: Float, clipping: Float)).ptr
  }

  def RNNForwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
    src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
    bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.RNNForwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
        src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
        bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long)).ptr
  }

  def RNNBackwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
    src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
    bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long, diff_src_layer_desc: Long,
    diff_src_iter_desc: Long, diff_weights_layer_desc: Long, diff_weights_iter_desc: Long,
    diff_bias_desc: Long, diff_dst_layer_desc: Long, diff_dst_iter_desc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      MklDnn.RNNBackwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
        src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
        bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long, diff_src_layer_desc: Long,
        diff_src_iter_desc: Long, diff_weights_layer_desc: Long, diff_weights_iter_desc: Long,
        diff_bias_desc: Long, diff_dst_layer_desc: Long, diff_dst_iter_desc: Long)).ptr
  }

  def ReorderPrimitiveDescCreate(input: Long, output: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.ReorderPrimitiveDescCreate(input, output)).ptr
  }

  def ReorderPrimitiveDescCreateV2(input: Long, output: Long, attr: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.ReorderPrimitiveDescCreateV2(input, output, attr)).ptr
  }

  def PrimitiveCreate0(desc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitive(
      MklDnn.PrimitiveCreate0(desc)).ptr
  }

  def PrimitiveCreate2(desc: Long, inputs: Array[Long], indexes: Array[Int], inputLen: Int,
    outputs: Array[Long], outputLen: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitive(
      MklDnn.PrimitiveCreate2(desc: Long, inputs: Array[Long], indexes: Array[Int], inputLen: Int,
        outputs: Array[Long], outputLen: Int)).ptr
  }

  def PrimitiveDescCreate(opDesc: Long, engine: Long, hingForwardPrimitiveDesc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(MklDnn.PrimitiveDescCreate(opDesc, engine, hingForwardPrimitiveDesc)).ptr
  }

  def PrimitiveDescCreateV2(opDesc: Long, attr: Long, engine: Long,
    hingForwardPrimitiveDesc: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.PrimitiveDescCreateV2(opDesc: Long, attr: Long, engine: Long,
        hingForwardPrimitiveDesc: Long)).ptr
  }

  def MemoryPrimitiveDescCreate(desc: Long, engine: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.MemoryPrimitiveDescCreate(desc, engine)).ptr
  }

  def ConcatPrimitiveDescCreate(output_desc: Long, n: Int, concat_dimension: Int,
    input_pds: Array[Long])(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.ConcatPrimitiveDescCreate(output_desc: Long, n: Int, concat_dimension: Int,
      input_pds: Array[Long])).ptr
  }

  def ViewPrimitiveDescCreate(memory_primitive_desc: Long, dims: Array[Int], offsets: Array[Int])
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.ViewPrimitiveDescCreate(memory_primitive_desc: Long, dims: Array[Int], offsets: Array[Int])).ptr
  }

  def SumPrimitiveDescCreate(output_mem_desc: Long, n: Int, scales: Array[Float],
    input_pds: Array[Long])(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      MklDnn.SumPrimitiveDescCreate(output_mem_desc: Long, n: Int, scales: Array[Float],
        input_pds: Array[Long])).ptr
  }

  def CreateAttr()(implicit owner: MemoryOwner): Long = {
    new MklMemoryAttr(
      MklDnn.CreateAttr()).ptr
  }
  def CreatePostOps()(implicit owner: MemoryOwner): Long = {
    new MklMemoryPostOps(
      MklDnn.CreatePostOps()).ptr
  }
// scalastyle:on
}
