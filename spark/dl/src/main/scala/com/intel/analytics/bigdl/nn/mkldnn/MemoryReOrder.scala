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
 * when from mkldnn layer to bigdl layer, there need to do reorder for input
 * @param format
 */
class MemoryReOrder(format: DataFormat = DataFormat.NHWC) extends TensorModule[Float] {

  @transient
  private val engine = this.getDnnEngine(0)
  @transient
  private val stream = this.getStream()

  this.output_format = this.format.value match {
    case "NHWC" => MklDnn.MemoryFormat.nhwc
    case "NCHW" => MklDnn.MemoryFormat.nchw
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    require(input.getPrimitiveDesc() != 0L,
      "input must be from mkldnn layer, and have a primitive describer")
    val input_sizes = input.size()
    val dim = input.dim()
    output.resize(input_sizes)

    val dst_memory = MklDnnOps.initDataMemory(
                      dim, input_sizes, this.output_format, MklDnn.DataType.f32, engine, output)
    val src_pd = input.getPrimitiveDesc()
    val (reorder_primitive, src_memory) = MklDnnOps.prepareReorder(dst_memory, src_pd, false)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(input, output)
    val stream_fwd = Array(reorder_primitive)
    val n_fwd = stream_fwd.length
    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd, n_fwd, memoryPrimitives, buffer)
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    gradOutput
  }
}

object MemoryReOrder {
  def apply[T: ClassTag](format: DataFormat = DataFormat.NCHW): MemoryReOrder = {
    new MemoryReOrder(format)
  }
}