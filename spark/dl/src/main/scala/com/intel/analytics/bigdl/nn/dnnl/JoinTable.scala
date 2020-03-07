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

import com.intel.analytics.bigdl.dnnl.{DNNL, Memory, ArgType}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


class JoinTable(val dimension: Int) extends MklDnnLayer {

  @transient
  private var bwdSrcMds: Array[Long] = null
  @transient
  private var bwdDstMds: Array[Long] = null

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs.length > 0, s"at least one tensor, but is ${inputs.length}")
    _inputFormats = nativeData(inputs)

    val totalShape = inputs(0).shape.clone()
    var i = 1
    while(i < inputs.length) {
      val curShape = inputs(i).shape
      require(totalShape.length == curShape.length, "tensor dimension not match")
      // require(inputs(i).isInstanceOf[NativeData], "memory should be native")
      var j = 0
      while(j < curShape.length) {
        if (j == dimension - 1) {
          totalShape(j) += curShape(j)
        } else {
          require(totalShape(j) == curShape(j), "tensor size not match")
        }
        j += 1
      }

      _inputFormats(i) = MemoryData.cloneFormatWithDesc(inputs(i))
      i += 1
    }

    val dstMd = DnnlMemory.MemoryDescInit(totalShape.length,
      totalShape, inputs(0).dataType, Memory.FormatTag.any)

    val primDesc = DnnlMemory.ConcatPrimitiveDescCreate(
      dstMd,
      inputs.length,
      dimension - 1,
      _inputFormats.map(_.getMemoryDescriptor()),
      runtime.engine
    )

    _outputFormats = Array(MemoryData.primitiveOutput(primDesc))
    _outputFormats.head.getMemoryObject(runtime)
    updateOutputPrimitives = Array(DnnlMemory.PrimitiveCreate(primDesc))
    output = initTensor(_outputFormats(0))

    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_DST -> outputFormats().head.getMemoryObject(runtime)
    )

    var idx = 0
    while (idx < inputs.length) {
      fwdExecArgs(ArgType.DNNL_ARG_MULTIPLE_SRC + idx) =
        inputs(idx).getMemoryObject(runtime)
      idx += 1
    }

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grads: Array[MemoryData], phase: Phase) = {
    _gradOutputFormats = singleNativeData(grads)
    _gradOutputFormatsForWeight = _gradOutputFormats
    _gradInputFormats = _inputFormats.map(f => {
      MemoryData.cloneFormatWithDesc(f)
    })
    val prims = new ArrayBuffer[Long]()
    val srcMds = new ArrayBuffer[Long]()
    val dstMds = new ArrayBuffer[Long]()

    val offset = new Array[Int](_gradOutputFormats(0).shape.length)

    for(i <- 0 until _gradInputFormats.length) {
      val currSrcMd = DNNL.InitSubmemory(
        _gradOutputFormats(0).getMemoryDescriptor(),
        _gradInputFormats(i).shape.map(_.toLong), offset.map(_.toLong))

      val currDstMd = gradInputFormats()(i).getMemoryDescriptor()

      val reorderPd = DnnlMemory.ReorderPrimitiveDescCreate(
        currSrcMd,
        currDstMd,
        runtime.engine,
        0L)

      val reorderPrim = DnnlMemory.PrimitiveCreate(reorderPd)
      prims.append(reorderPrim)
      srcMds.append(currSrcMd)
      dstMds.append(currDstMd)

      offset(dimension - 1) += _gradInputFormats(i).shape(dimension - 1)
    }
    updateGradInputPrimitives = prims.toArray
    bwdSrcMds = srcMds.toArray
    bwdDstMds = dstMds.toArray
    gradInput = initActivity(_gradInputFormats)

    (_gradOutputFormats, _gradInputFormats)
  }


  override def updateOutput(input: Activity): Activity = {
    assert(input.isTable && output.isTensor)
    val inputSrcs = input.toTable

    updateOutputTensors = mutable.Map(
      ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
    )

    var idx = 0
    while (idx < inputSrcs.length()) {
      updateOutputTensors(ArgType.DNNL_ARG_MULTIPLE_SRC + idx) =
        inputSrcs(idx + 1).asInstanceOf[Tensor[Float]]
      idx += 1
    }

    MklDnnOps.streamSubmit(updateOutputPrimitives,
      runtime.stream, fwdExecArgs,
      updateOutputTensors
    )

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    require(gradOutput.isTensor, "gradOutput should be tensor")
    require(gradInput.isTable, "gradInput should be table")
    val _gradOutput = gradOutput.asInstanceOf[Tensor[Float]]
    val _gradInput = gradInput.toTable
    val length = _gradInput.length()
    require(length == updateGradInputPrimitives.length, "gradOutput number not match")
    var i = 0
    while(i < length) {
      bwdExecArgs = mutable.Map(
        ArgType.DNNL_ARG_FROM -> DNNL.MemoryCreate(bwdSrcMds(i), runtime.engine),
        ArgType.DNNL_ARG_TO -> DNNL.MemoryCreate(bwdDstMds(i), runtime.engine)
      )
      updateGradInputTensors = mutable.Map(
        ArgType.DNNL_ARG_FROM -> gradOutput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_TO ->  _gradInput[Tensor[Float]](i + 1)
      )
      MklDnnOps.streamSubmit(Array(updateGradInputPrimitives(i)), runtime.stream,
        bwdExecArgs, updateGradInputTensors)
      i += 1
    }

    gradInput
  }

}

object JoinTable {
  def apply(dimension: Int): JoinTable = new JoinTable(dimension)
}
