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

import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.{Utils => NNUtils}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, DataFormat}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor

class AvgPooling(
  var kW: Int,
  var kH: Int,
  dW: Int = 1,
  dH: Int = 1,
  padW: Int = 0,
  padH: Int = 0,
  globalPooling: Boolean = false,
  val format: DataFormat = DataFormat.NCHW
) extends MklDnnLayer {
  @transient private var paddingTL: Array[Int] = _
  @transient private var paddingBR: Array[Int] = _
  @transient private var fwdPD: Long = _

  // reminder: ceilMode default value is true,
  // but in blas SpatialMaxPooling, default ceilMode is false
  private var ceilMode = true

  /**
   * set ceil mode
   * @return this
   */
  def ceil(): AvgPooling = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): AvgPooling = {
    ceilMode = false
    this
  }

  private val algKind = if (padH == -1 && padW == -1) {
    AlgKind.PoolingAvgIncludePadding
  } else {
    AlgKind.PoolingAvgExcludePadding
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    _inputFormats = singleNativeData(inputs)
    val n = _inputFormats(0).shape(0)
    val c = _inputFormats(0).shape(1)
    val h = _inputFormats(0).shape(2)
    val w = _inputFormats(0).shape(3)

    // global average pooling reduce each feature map to a single average value
    if (globalPooling) {
      kH = h
      kW = w
    }

    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)

    val (pt, pb, pl, pr, oh, ow) = if (padH == -1 && padW == -1) {
      val sizes = NNUtils.getSAMEOutSizeAndPadding(h, w, dH, dW, kH, kW)
      (sizes(0), sizes(1), sizes(2), sizes(3), sizes(4), sizes(5))
    } else {
      NNUtils.getPaddingAndOutputSize(h, w, dH, dW, kH, kW, padH, padW, ceilMode)
    }

    paddingTL = Array(pt, pl)
    paddingBR = Array(pb, pr)
    val outputMD = MklDnnMemory.MemoryDescInit(4, Array(n, c, oh, ow), inputs(0).dataType,
      Memory.Format.any)

    val kind = if (phase == InferencePhase) {
      PropKind.ForwardScoring
    } else {
      PropKind.ForwardTraining
    }

    val description = MklDnnMemory.PoolingForwardDescInit(
      kind, algKind,
      _inputFormats(0).getMemoryDescription(), outputMD, strides, kernel, paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)
    fwdPD = MklDnnMemory.PrimitiveDescCreate(description, runtime.engine, 0L)
    _outputFormats = Array(MemoryData.primitiveOutput(fwdPD))
    output = initTensor(_outputFormats(0))
    updateOutputPrimitives = Array(MklDnnMemory.PrimitiveCreate2(fwdPD,
      _inputFormats.map(_.getPrimitive(runtime)), Array(0), 1,
      _outputFormats.map(_.getPrimitive(runtime)), 2))
    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grad)
    _gradOutputFormatsForWeight = _gradOutputFormats
    val strides = Array(dW, dH)
    val kernel = Array(kH, kW)
    val description = MklDnnMemory.PoolingBackwardDescInit(algKind,
      _inputFormats(0).getMemoryDescription(),
      _gradOutputFormats(0).getMemoryDescription(),
      strides, kernel, paddingTL, paddingBR, MklDnn.PaddingKind.mkldnnPaddingZero)

    val pd = MklDnnMemory.PrimitiveDescCreate(description, runtime.engine, fwdPD)
    _gradInputFormats = Array(MemoryData.operationWant(pd, Query.DiffSrcPd))
    updateGradInputPrimitives = Array(MklDnnMemory.PrimitiveCreate2(pd,
      _gradOutputFormats.map(_.getPrimitive(runtime)),
      Array(0, 0), 2, _gradInputFormats.map(_.getPrimitive(runtime)), 1))
    gradInput = initTensor(_gradInputFormats(0))
    (_gradOutputFormats, _gradInputFormats)
  }
}

object AvgPooling {
  def apply(
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    globalPooling: Boolean = false,
    format: DataFormat = DataFormat.NCHW
  ): AvgPooling = new AvgPooling(kW, kH, dW, dH, padW, padH, globalPooling, format = format)
}
