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
class MemoryReOrder(inputFormat: Int = MklDnn.MemoryFormat.any,
                    outputFormat: Int = MklDnn.MemoryFormat.nchw) extends TensorModule[Float] {

  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L
  @transient
  private var update_primitive: Boolean = true
  @transient
  private var inputElement : Int = 0
  @transient
  private var src_memory: Long = 0L
  @transient
  private var src_memory2: Long = 0L
  @transient
  private var dst_memory: Long = 0L
  @transient
  private var dst_memory2: Long = 0L
  @transient
  private var gradInput_memory: Long = 0L
  @transient
  private var gradOutput_memory: Long = 0L

  override val isMklDnnModel: Boolean = true

  val stream_fwd = new ArrayBuffer[Long]
  val stream_bwd = new ArrayBuffer[Long]

  private val dataType = MklDnn.DataType.f32

  private var internal_inputFormat = this.inputFormat

  require(outputFormat != MklDnn.MemoryFormat.any,
          "output format in MemoryReOrder should not be any")

  // convert input from input format to output format
  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    if (inputElement != input.nElement()) {
      update_primitive = true
      inputElement = input.nElement()
    } else {
      update_primitive = false
    }

    if (update_primitive) {
      val sizes = input.size()
      val dim = input.dim()
      output.resizeAs(input)

      val src_pd = if (input.getPrimitiveDesc() != 0L) {
        internal_inputFormat = input.getFormat()
        input.getPrimitiveDesc()
      } else {
        require(inputFormat != MklDnn.MemoryFormat.any,
                "input format in MemoryReOrder should not be any")
        val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, internal_inputFormat)
        MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
      }

      val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, this.outputFormat)
      val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
      dst_memory = MklDnn.PrimitiveCreate0(user_pd)

      val res = MklDnnOps.prepareReorder(dst_memory, src_pd, false)
      // val reorder_primitive = res._1
      src_memory = res._2
      if (src_memory != 0L) {
        output.setPrimitiveDesc(user_pd)
      }

      stream_fwd.clear()
      stream_fwd.append(res._1)
    }

    /* build a simple net */
    if (src_memory != 0L) {
      val memoryPrimitives = Array(src_memory, dst_memory)
      val buffer = Array(input, output)
      MklDnnOps.streamSubmit(stream, 1, stream_fwd.toArray, 1, memoryPrimitives, buffer)
    } else {
      output = input
    }
    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      println(s"MemoryReorderForGradoutput dnn ${this.getName()} forward ${end1}")
    }
    output
  }

  // convert gradOutput from output format to input format
  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()

    if (update_primitive) {
      val sizes = gradOutput.size()
      val dim = gradOutput.dim()
      gradInput.resizeAs(gradOutput)

      val src_pd = if (gradOutput.getPrimitiveDesc() != 0L) {
        gradOutput.getPrimitiveDesc()
      } else {
        val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, this.outputFormat)
        MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
      }

      val prim_md = MklDnn.MemoryDescInit(dim, sizes, dataType, internal_inputFormat)
      val user_pd = MklDnn.MemoryPrimitiveDescCreate(prim_md, engine)
      dst_memory2 = MklDnn.PrimitiveCreate0(user_pd)

      val res = MklDnnOps.prepareReorder(dst_memory2, src_pd, false)
      // val reorder_primitive = res._1
      src_memory2 = res._2
      if (src_memory2 != 0L) {
        gradInput.setPrimitiveDesc(user_pd)
      }

      stream_bwd.clear()
      stream_bwd.append(res._1)
    }

    /* build a simple net */
    if (src_memory2 != 0L) {
      val memoryPrimitives = Array(src_memory2, dst_memory2)
      val buffer = Array(gradOutput, gradInput)
      MklDnnOps.streamSubmit(stream, 1, stream_bwd.toArray, 1, memoryPrimitives, buffer)
    } else {
      gradInput = gradOutput
    }

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      println(s"MemoryReorderForGradoutput dnn ${this.getName()} backward ${end1}")
    }
    gradInput
  }
}

object MemoryReOrder {
  def apply[T: ClassTag](inputFormat: Int = MklDnn.MemoryFormat.any,
      outputFormat: Int = MklDnn.MemoryFormat.nchw): MemoryReOrder = {
    new MemoryReOrder(inputFormat, outputFormat)
  }
}
