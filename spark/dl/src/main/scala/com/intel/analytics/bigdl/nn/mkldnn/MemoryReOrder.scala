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

import breeze.linalg
import breeze.linalg.dim
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * when from mkldnn layer to bigdl layer, there need to do reorder for input or gradOutput
 */
class MemoryReOrder(inputFormat: Int = MklDnn.MemoryFormat.nchw,
                    outputFormat: Int = MklDnn.MemoryFormat.any) extends TensorModule[Float] {

  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L

  override val isMklDnnModel: Boolean = true

  private val dataType = MklDnn.DataType.f32

  // convert input
  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    val sizes = input.size()
    val dim = input.dim()
    output.resizeAs(input)

    val dst_memory = MklDnnOps.initDataMemory(dim, sizes, this.outputFormat, dataType, engine)
    val src_pd = if (input.getPrimitiveDesc() != 0L) {
      input.getPrimitiveDesc()
    } else {
      val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, this.inputFormat)
      MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
    }
    val (reorder_primitive, src_memory) = MklDnnOps.prepareReorder(dst_memory, src_pd, false)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(input, output)
    val stream_fwd = Array(reorder_primitive)
    val n_fwd = stream_fwd.length
    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd, n_fwd, memoryPrimitives, buffer)
    output
  }

  // convert gradOutput
  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val sizes = gradOutput.size()
    val dim = gradOutput.dim()
    gradInput.resizeAs(gradOutput)

    val dst_memory = MklDnnOps.initDataMemory(dim, sizes, this.outputFormat, dataType, engine)
    val src_pd = if (gradOutput.getPrimitiveDesc() != 0L) {
      gradOutput.getPrimitiveDesc()
    } else {
      val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, this.inputFormat)
      MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
    }
    val (reorder_primitive, src_memory) = MklDnnOps.prepareReorder(dst_memory, src_pd, false)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(gradOutput, gradInput)
    val stream_fwd = Array(reorder_primitive)
    val n_fwd = stream_fwd.length
    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd, n_fwd, memoryPrimitives, buffer)

    gradInput
  }
}

object MemoryReOrder {
  def apply[T: ClassTag](inputFormat: Int = MklDnn.MemoryFormat.nchw,
                         outputFormat: Int = MklDnn.MemoryFormat.nhwc): MemoryReOrder = {
    new MemoryReOrder(inputFormat, outputFormat)
  }
}
