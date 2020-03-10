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

import com.intel.analytics.bigdl.dnnl.DNNL



// TODO: rename to DNNL
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
  def doRelease(): Unit = DNNL.PrimitiveDescDestroy(ptr)
}

class MklMemoryAttr(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = DNNL.DestroyAttr(ptr)
}

class MklMemoryPostOps(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = DNNL.DestroyPostOps(ptr)
}

// All *DescInit memory objects share the same dealloactor
class MklMemoryDescInit(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = DNNL.FreeMemoryDescInit(ptr)
}

class MklMemoryPrimitive(_ptr: Long)(implicit owner: MemoryOwner)
  extends MklDnnNativeMemory(_ptr) {
  def doRelease(): Unit = DNNL.PrimitiveDestroy(ptr)
}


object DnnlMemory {

  // scalastyle:off
  def MemoryDescInit(ndims: Int, dims: Array[Int], dataType: Int, dataFormat: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.MemoryDescInit(ndims, dims.map(_.toLong), dataType, dataFormat)).ptr
  }

  def MemoryDescInitByStrides(ndims: Int, dims: Array[Int], dataType: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.MemoryDescInitByStrides(ndims, dims.map(_.toLong), dataType)).ptr
  }

  def EltwiseForwardDescInit(propKind: Int, algKind: Int, srcDesc: Long, alpha: Float,
    beta: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.EltwiseForwardDescInit(propKind, algKind, srcDesc, alpha, beta)).ptr
  }

  def EltwiseBackwardDescInit(algKind: Int, diffDataDesc: Long, dataDesc: Long, alpha: Float,
    beta: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.EltwiseBackwardDescInit(algKind, diffDataDesc, dataDesc, alpha, beta)).ptr
  }

  def LinearForwardDescInit(propKind: Int, srcMemDesc: Long, weightMemDesc: Long,
    biasMemDesc: Long, dstMemDesc: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.LinearForwardDescInit(propKind, srcMemDesc, weightMemDesc, biasMemDesc, dstMemDesc)).ptr
  }

  def LinearBackwardDataDescInit(diffSrcMemDesc: Long, weightMemDesc: Long, diffDstMemDesc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc, diffDstMemDesc)).ptr
  }

  def LinearBackwardWeightsDescInit(srcMemDesc: Long, diffWeightMemDesc: Long,
    diffBiasMemDesc: Long, diffDstMemDesc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.LinearBackwardWeightsDescInit(srcMemDesc,
        diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc)).ptr
  }

  def BatchNormForwardDescInit(propKind: Int, srcMemDesc: Long, epsilon: Float, flags: Long)
  (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.BatchNormForwardDescInit(propKind, srcMemDesc, epsilon, flags)).ptr
  }

  def BatchNormBackwardDescInit(prop_kind: Int, diffDstMemDesc: Long, srcMemDesc: Long,
    epsilon: Float, flags: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.BatchNormBackwardDescInit(prop_kind, diffDstMemDesc, srcMemDesc, epsilon, flags)).ptr
  }

  def SoftMaxForwardDescInit(prop_kind: Int, dataDesc: Long, axis: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.SoftMaxForwardDescInit(prop_kind, dataDesc, axis)).ptr
  }

//  def ConvForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, weights_desc: Long,
//    bias_desc: Long, dst_desc: Long, strides: Array[Int], padding_l: Array[Int],
//    padding_r: Array[Int], padding_kind: Int)(implicit owner: MemoryOwner): Long = {
//
////    public native static long ConvForwardDescInit(int prop_kind, int alg_kind,
////      long src_desc, long weights_desc,
////      long bias_desc, long dst_desc,
////      long[] strides, long[] padding_l,
////      long[] padding_r);
//    new MklMemoryDescInit(
//      DNNL.ConvForwardDescInit(prop_kind, alg_kind, src_desc, weights_desc,
//        bias_desc, dst_desc, strides.map(_.toLong), padding_l.map(_.toLong),
//        padding_r.map(_.toLong))).ptr
//
//    0L
//  }

  def DilatedConvForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long,
    weights_desc: Long, bias_desc: Long, dst_desc: Long, strides: Array[Int],
    dilates: Array[Int], padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.DilatedConvForwardDescInit(prop_kind, alg_kind, src_desc,
        weights_desc, bias_desc, dst_desc, strides.map(_.toLong),
        dilates.map(_.toLong), padding_l.map(_.toLong), padding_r.map(_.toLong))).ptr
  }

//  def ConvBackwardWeightsDescInit(alg_kind: Int, src_desc: Long, diff_weights_desc: Long,
//    diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int],
//    padding_r: Array[Int], padding_kind: Int)(implicit owner: MemoryOwner): Long = {
//    new MklMemoryDescInit(
//      DNNL.ConvBackwardWeightsDescInit(alg_kind, src_desc, diff_weights_desc,
//        diff_bias_desc, diff_dst_desc, strides.map(_.toLong), padding_l.map(_.toLong),
//        padding_r.map(_.toLong))).ptr
//  }

  def DilatedConvBackwardWeightsDescInit(alg_kind: Int, src_desc: Long, diff_weights_desc: Long,
    diff_bias_desc: Long, diff_dst_desc: Long, strides: Array[Int], dilates: Array[Int],
    padding_l: Array[Int], padding_r: Array[Int], padding_kind: Int)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.DilatedConvBackwardWeightsDescInit(alg_kind, src_desc,
        diff_weights_desc,
        diff_bias_desc, diff_dst_desc, strides.map(_.toLong), dilates.map(_.toLong),
        padding_l.map(_.toLong), padding_r.map(_.toLong))).ptr
  }

//  def ConvBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
//    diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
//    padding_kind: Int)(implicit owner: MemoryOwner): Long = {
//    new MklMemoryDescInit(
//      DNNL.ConvBackwardDataDescInit(alg_kind, diff_src_desc, weights_desc, diff_dst_desc,
//        strides.map(_.toLong), padding_l.map(_.toLong), padding_r.map(_.toLong))).ptr
//  }

  def DilatedConvBackwardDataDescInit(alg_kind: Int, diff_src_desc: Long, weights_desc: Long,
    diff_dst_desc: Long, strides: Array[Int], padding_l: Array[Int], dilates: Array[Int],
    padding_r: Array[Int], padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.DilatedConvBackwardDataDescInit(alg_kind, diff_src_desc, weights_desc, diff_dst_desc,
        strides.map(_.toLong), padding_l.map(_.toLong), dilates.map(_.toLong), padding_r.map(_.toLong))).ptr
  }

  def PoolingForwardDescInit(prop_kind: Int, alg_kind: Int, src_desc: Long, dst_desc: Long,
    strides: Array[Int], kernel: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.PoolingForwardDescInit(prop_kind, alg_kind, src_desc, dst_desc,
        strides.map(_.toLong), kernel.map(_.toLong), padding_l.map(_.toLong), padding_r.map(_.toLong))).ptr
  }

  def PoolingBackwardDescInit(alg_kind: Int, diff_src_desc: Long, diff_dst_desc: Long,
    strides: Array[Int], kernel: Array[Int], padding_l: Array[Int], padding_r: Array[Int],
    padding_kind: Int)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.PoolingBackwardDescInit(alg_kind, diff_src_desc, diff_dst_desc,
        strides.map(_.toLong), kernel.map(_.toLong), padding_l.map(_.toLong), padding_r.map(_.toLong))).ptr
  }

  def LRNForwardDescInit(prop_kind: Int, alg_kind: Int, data_desc: Long, local_size: Int,
    alpha: Float, beta: Float, k: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.LRNForwardDescInit(prop_kind, alg_kind, data_desc, local_size,
        alpha, beta, k)).ptr
  }

  def LRNBackwardDescInit(alg_kind: Int, diff_data_desc: Long, data_desc: Long, local_size: Int,
    alpha: Float, beta: Float, k: Float)(implicit owner: MemoryOwner): Long = {
    new MklMemoryDescInit(
      DNNL.LRNBackwardDescInit(alg_kind: Int, diff_data_desc: Long, data_desc: Long,
        local_size: Int, alpha: Float, beta: Float, k: Float)).ptr
  }

  def RNNCellDescInit(kind: Int, f: Int, flags: Int, alpha: Float, clipping: Float)
    (implicit owner: MemoryOwner): Long = {
    // TODO
//    new MklMemoryDescInit(
//      DNNL.RNNCellDescInit(kind: Int, f: Int, flags: Int, alpha: Float, clipping: Float)).ptr
    0L
  }

  def RNNForwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
    src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
    bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long)
    (implicit owner: MemoryOwner): Long = {
    // TODO

//    JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_VanillaRNNForwardDescInit(
//      JNIEnv *env, jclass cls,
//      int prop_kind, int activation_kind,
//      int direction, long src_layer_desc,
//      long src_iter_desc, long weights_layer_desc,
//      long weights_iter_desc, long bias_desc,
//      long dst_layer_desc, long dst_iter_desc, unsigned int flags,
//      float alpha, float beta)
    0L
  }

  def RNNBackwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
    src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
    bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long, diff_src_layer_desc: Long,
    diff_src_iter_desc: Long, diff_weights_layer_desc: Long, diff_weights_iter_desc: Long,
    diff_bias_desc: Long, diff_dst_layer_desc: Long, diff_dst_iter_desc: Long)
    (implicit owner: MemoryOwner): Long = {
    // TODO
//    new MklMemoryDescInit(
//      DNNL.RNNBackwardDescInit(prop_kind: Int, rnn_cell_desc: Long, direction: Int,
//        src_layer_desc: Long, src_iter_desc: Long, weights_layer_desc: Long, weights_iter_desc: Long,
//        bias_desc: Long, dst_layer_desc: Long, dst_iter_desc: Long, diff_src_layer_desc: Long,
//        diff_src_iter_desc: Long, diff_weights_layer_desc: Long, diff_weights_iter_desc: Long,
//        diff_bias_desc: Long, diff_dst_layer_desc: Long, diff_dst_iter_desc: Long)).ptr
    0L
  }

  def ReorderPrimitiveDescCreate(input: Long, output: Long, engine: Long, attr: Long)
    (implicit owner: MemoryOwner): Long = {
    // TODO:
    new MklMemoryPrimitiveDesc(
      DNNL.ReorderPrimitiveDescCreate(input, engine, output, engine, attr)).ptr
  }

//  def ReorderPrimitiveDescCreateV2(input: Long, output: Long, attr: Long)
//    (implicit owner: MemoryOwner): Long = {
//    // TODO
////    new MklMemoryPrimitiveDesc(
////      DNNL.ReorderPrimitiveDescCreateV2(input, output, attr)).ptr
//    0L
//  }

  def PrimitiveCreate(desc: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitive(DNNL.PrimitiveCreate(desc)).ptr
  }

  def PrimitiveDescCreate(opDesc: Long, engine: Long, hingForwardPrimitiveDesc: Long)
    (implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(DNNL.PrimitiveDescCreate(opDesc, engine, hingForwardPrimitiveDesc)).ptr
  }

  def PrimitiveDescCreateV2(opDesc: Long, attr: Long, engine: Long,
    hingForwardPrimitiveDesc: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      DNNL.PrimitiveDescCreateV2(opDesc: Long, attr: Long, engine: Long,
        hingForwardPrimitiveDesc: Long)).ptr
  }

  def MemoryPrimitiveDescCreate(desc: Long, engine: Long)
    (implicit owner: MemoryOwner): Long = {
    // TODO
//    new MklMemoryPrimitiveDesc(
//      DNNL.MemoryPrimitiveDescCreate(desc, engine)).ptr
    0L
  }

  def ConcatPrimitiveDescCreate(output_desc: Long, n: Int, concat_dimension: Int,
                                input_pds: Array[Long], engine: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      DNNL.ConcatPrimitiveDescCreate(output_desc: Long, n: Int, concat_dimension: Int,
        input_pds: Array[Long], 0L, engine)).ptr
  }

  def ViewPrimitiveDescCreate(memory_primitive_desc: Long, dims: Array[Int], offsets: Array[Int])
    (implicit owner: MemoryOwner): Long = {
    // TODO
//    new MklMemoryPrimitiveDesc(
//      DNNL.ViewPrimitiveDescCreate(memory_primitive_desc: Long, dims: Array[Int], offsets: Array[Int])).ptr
    0L
  }

  def SumPrimitiveDescCreate(output_mem_desc: Long, n: Int, scales: Array[Float],
    input_mds: Array[Long], attr: Long, engine: Long)(implicit owner: MemoryOwner): Long = {
    new MklMemoryPrimitiveDesc(
      DNNL.SumPrimitiveDescCreate(output_mem_desc, n, scales,
        input_mds, attr, engine)).ptr
  }

  def CreateAttr()(implicit owner: MemoryOwner): Long = {
    new MklMemoryAttr(
      DNNL.CreateAttr()).ptr
  }
  def CreatePostOps()(implicit owner: MemoryOwner): Long = {
    new MklMemoryPostOps(
      DNNL.CreatePostOps()).ptr
  }
// scalastyle:on
}
