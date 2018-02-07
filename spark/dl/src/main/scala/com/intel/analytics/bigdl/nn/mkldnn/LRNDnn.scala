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

import breeze.linalg.*
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.{SpatialCrossMapLRN, SpatialMaxPooling, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class LRNDnn[T: ClassTag](
   size: Int = 5,
   alpha: Double = 1.0,
   beta: Double = 0.75,
   k: Double = 1.0,
   format: DataFormat = DataFormat.NCHW)
   (implicit ev: TensorNumeric[T])extends TensorModule[Float] with Initializable {

  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L
  @transient
  private var inputElement : Int = 0
  @transient
  private var src_md: Long = 0L
  @transient
  private var src_memory: Long = 0L
  @transient
  private var dst_memory: Long = 0L
  @transient
  private var work_memory: Long = 0L
  @transient
  private var gradInput_memory: Long = 0L
  @transient
  private var gradOutput_memory: Long = 0L
  @transient
  private var fwd_pd: Long = 0L
  @transient
  private var pool_fwd: Long = 0L
  @transient
  private var pool_bwd: Long = 0L
  @transient
  private var update_primitive: Boolean = true
  @transient
  private var reorder_gradOutput_memory: Long = 0L

  val stream_fwd = new ArrayBuffer[Long]
  val stream_bwd = new ArrayBuffer[Long]
  val stream_acc = new ArrayBuffer[Long]

  var fwd : Long = 0L

  //
  private var input_format = MklDnn.MemoryFormat.nchw
  private val dataType = MklDnn.DataType.f32

  private var workSpace : MklDnnTensor[Float] = null
  private var inputBuffer : MklDnnTensor[Float] = null
  private var gradOutputBuffer : MklDnnTensor[Float] = null

  // test
  private var dst_pd: Long = 0L

  def reorderToInternal(user_md: Long, pd: Long, queryType: Int, data: Tensor[Float],
                        data_size: Array[Int], index: Int = 0): (Long, Long) = {
    val internal_pd = MklDnnOps.primitiveDescQueryPd(pd, queryType, index)
    val res = MklDnnOps.prepareReorder(user_md, internal_pd, true)
    if (res._1 != 0L) {
      data.setPrimitiveDesc(internal_pd)
    }
    res
  }


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

    if (inputBuffer == null || inputBuffer.nElement() < input.nElement() ||
      input.getTensorType != MklDnnType) {
      inputBuffer = MklDnnTensor[Float](input.size())
    }

    if (update_primitive) {
      input_format = input.dim() match {
        case 1 => MklDnn.MemoryFormat.x
        case 2 => MklDnn.MemoryFormat.nc
        case 4 => MklDnn.MemoryFormat.nchw
      }
      val nbatch = input.size(1)
      val input_size = input.size()
      val dst_sizes = input_size
      // todo: output with Dense Tensor
      output = MklDnnTensor[Float](dst_sizes)

      // create memory desc, for input
      if (input.getPrimitiveDesc() != 0L) {
        val input_pd = input.getPrimitiveDesc()
        src_memory = MklDnn.PrimitiveCreate0(input_pd)
        src_md = MklDnnOps.primitiveDescQueryMemory(input_pd)
      } else {
        src_md = MklDnnOps.memoryDescInit(input.dim(), input.size(), dataType, this.input_format)
        src_memory = MklDnnOps.createMemoryPrimitive(src_md, engine)
      }

      /* create a convolution */
      val lrn_desc = MklDnn.LRNForwardDescInit(
                        MklDnn.PropKind.forwardTraining, MklDnn.AlgKind.lrnAcrossChannels,
                        src_md, size, alpha.toFloat, beta.toFloat, k.toFloat)

      fwd_pd = MklDnnOps.primitiveDescCreate(lrn_desc, engine, 0L)

      /* create memory for dst data, we don't need to reorder it to user data */
      dst_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.dst_pd, 0)
      dst_memory = MklDnn.PrimitiveCreate0(dst_pd)
      output.setPrimitiveDesc(dst_pd)

      val workdspace_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.workspace_pd, 0)
      work_memory = MklDnn.PrimitiveCreate0(workdspace_pd)
      val workdspace_size = MklDnn.PrimitiveDescGetSize(workdspace_pd)
      // todo: support resize
      if (workSpace == null) workSpace = MklDnnTensor[Float](Array(workdspace_size.toInt))

      val inputs = Array(src_memory)
      val outputs = Array(dst_memory, work_memory)
      val indexes = Array(0)
      fwd = MklDnnOps.primitiveCreate2(fwd_pd, inputs, indexes, 1, outputs, 2)

      /* build a simple net */
      stream_fwd.clear()
      stream_fwd.append(fwd)
    }

    if (System.getProperty("debug") == "1") {
      println("lrn updateoutput " + this.getName())
    }
    val n_fwd = stream_fwd.length
    if (input.getTensorType != MklDnnType) {
      MklDnnTensor.syncFromHeap(inputBuffer, input.storage().array(), input.storageOffset() - 1)
    } else {
      inputBuffer = input.asInstanceOf[MklDnnTensor[Float]]
    }
    val memoryPrimitives = Array(src_memory, dst_memory, work_memory)
    val buffer = Array(inputBuffer, output, workSpace)
    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd.toArray, n_fwd, memoryPrimitives, buffer)

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      println(s"lrn dnn ${this.getName()} forward ${end1}")
    }
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    if (gradOutputBuffer == null || gradOutputBuffer.nElement() < gradOutput.nElement() ||
      gradOutput.getTensorType != MklDnnType) {
      gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
    }
    if (update_primitive) {
      var gradOutput_md : Long = 0L
      if (gradOutput.getPrimitiveDesc() != 0L) {
        val gradOutput_pd = gradOutput.getPrimitiveDesc()
        // gradOutput_md = MklDnnOps.primitiveDescQueryMemory(gradOutput_pd)
        gradOutput_md = MklDnn.PrimitiveDescQueryMemory(dst_pd)
        gradOutput_memory = MklDnn.PrimitiveCreate0(gradOutput_pd)
      } else {
        gradOutput_md = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
          MklDnn.DataType.f32, this.input_format)
        gradOutput_memory = MklDnnOps.createMemoryPrimitive(gradOutput_md, engine)
      }

      /* create backward descriptor */
      val bwd_desc = MklDnn.LRNBackwardDescInit(MklDnn.AlgKind.lrnAcrossChannels, src_md,
                            gradOutput_md, size, alpha.toFloat, beta.toFloat, k.toFloat)
      val bwd_pd = MklDnnOps.primitiveDescCreate(bwd_desc, engine, fwd_pd)

      /* create memory primities for relu gradInput */
      gradInput.resizeAs(input)
      val gradInput_pd = MklDnnOps.primitiveDescQueryPd(bwd_pd, MklDnn.Query.diff_src_pd, 0)
      gradInput_memory = MklDnn.PrimitiveCreate0(gradInput_pd)
      gradInput.setPrimitiveDesc(gradInput_pd)
      // gradInput.setPrimitiveDesc(MklDnn.PrimitiveDescQueryMemory(dst_pd))

      var reorder_gradOutput: Long = 0L
      val res = reorderToInternal(gradOutput_memory, bwd_pd, MklDnn.Query.diff_dst_pd,
        gradOutputBuffer, gradOutput.size())
      reorder_gradOutput = res._1
      reorder_gradOutput_memory = res._2

      val internal_gradOutput_memory = if (reorder_gradOutput_memory == 0L) {
        gradOutput_memory
      } else {
        println("lrn updateGradInput reorder")
        reorder_gradOutput_memory
      }

      val inputs = Array(src_memory, internal_gradOutput_memory, work_memory)
      val outputs = Array(gradInput_memory)
      val indexes = Array(0, 0, 0)
      val bwd = MklDnnOps.primitiveCreate2(bwd_pd, inputs, indexes, 3, outputs, 1)

      /* build a simple net */
      stream_bwd.clear()
      if (reorder_gradOutput_memory != 0L) stream_bwd.append(reorder_gradOutput)
      stream_bwd.append(bwd)
    }

    if (System.getProperty("debug") == "1") {
      println("lrn backward " + this.getName())
    }
    val n_bwd = stream_bwd.length
    val (memoryPrimitives, buffer) =
      if (reorder_gradOutput_memory == 0L && gradOutput.getTensorType != MklDnnType) {
        // sync here
        MklDnnTensor.syncFromHeap(
          gradOutputBuffer, gradOutput.storage().array(), gradOutput.storageOffset() - 1)
        (Array(src_memory, gradOutput_memory, work_memory, gradInput_memory),
          Array(inputBuffer, gradOutputBuffer, workSpace, gradInput))

      } else {
        (Array(src_memory, gradOutput_memory, work_memory, gradInput_memory,
          reorder_gradOutput_memory),
          Array(inputBuffer, gradOutput, workSpace, gradInput, gradOutputBuffer))
      }
    MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd, memoryPrimitives, buffer)

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      println(s"lrn dnn ${this.getName()} backward ${end1} format " + input.getFormat())
    }
    gradInput
  }

  override def clearState() : this.type = {
    super.clearState()
    gradOutputBuffer.set()
    workSpace.set()
    inputBuffer.set()
    this
  }
}

object LRNDnn {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: Int = 5,
    alpha: Double = 1.0,
    beta: Double = 0.75,
    k: Double = 1.0,
    format: DataFormat = DataFormat.NCHW)
  (implicit ev: TensorNumeric[T]) : LRNDnn[T] = {
    new LRNDnn[T](size, alpha, beta, k, format)
  }
}
