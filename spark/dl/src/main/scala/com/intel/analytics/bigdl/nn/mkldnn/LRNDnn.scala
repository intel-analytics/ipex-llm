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
import com.intel.analytics.bigdl.tensor.Tensor
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

  val stream_fwd = new ArrayBuffer[Long]
  val stream_bwd = new ArrayBuffer[Long]
  val stream_acc = new ArrayBuffer[Long]

  //
  private var input_format = MklDnn.MemoryFormat.nchw
  private val dataType = MklDnn.DataType.f32

  private var workSpace = Tensor[Float]()

  // test
  private var dst_pd: Long = 0L


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
      input_format = input.dim() match {
        case 1 => MklDnn.MemoryFormat.x
        case 2 => MklDnn.MemoryFormat.nc
        case 4 => MklDnn.MemoryFormat.nchw
      }
      val nbatch = input.size(1)
      val input_size = input.size()
      val dst_sizes = input_size
      output.resize(dst_sizes)

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
                        MklDnn.PropKind.forward, MklDnn.AlgKind.lrnAcrossChannels,
                        src_md, size, alpha.toFloat, beta.toFloat, k.toFloat)

      fwd_pd = MklDnnOps.primitiveDescCreate(lrn_desc, engine, 0L)

      /* create memory for dst data, we don't need to reorder it to user data */
      dst_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.dst_pd, 0)
      dst_memory = MklDnn.PrimitiveCreate0(dst_pd)
      output.setPrimitiveDesc(dst_pd)

      val workdspace_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.workspace_pd, 0)
      work_memory = MklDnn.PrimitiveCreate0(workdspace_pd)
      val workdspace_size = MklDnn.PrimitiveDescGetSize(workdspace_pd)
      workSpace.resize(workdspace_size.toInt)

      val inputs = Array(src_memory)
      val outputs = Array(dst_memory, work_memory)
      val indexes = Array(0)
      val fwd = MklDnnOps.primitiveCreate2(fwd_pd, inputs, indexes, 1, outputs, 2)

      /* build a simple net */
      stream_fwd.clear()
      stream_fwd.append(fwd)
    }
    val n_fwd = stream_fwd.length
    val memoryPrimitives = Array(src_memory, dst_memory, work_memory)
    val buffer = Array(input, output, workSpace)
    MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd.toArray, n_fwd, memoryPrimitives, buffer)

    val end1 = (System.nanoTime() - s1)/1e9
    // println(s"lrn dnn forward ${end1}")
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    if (update_primitive) {
      var gradOutput_md : Long = 0L
      if (gradOutput.getPrimitiveDesc() != 0L) {
        val gradOutput_pd = gradOutput.getPrimitiveDesc()
        gradOutput_md = MklDnnOps.primitiveDescQueryMemory(gradOutput_pd)
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

      val inputs = Array(src_memory, gradOutput_memory, work_memory)
      val outputs = Array(gradInput_memory)
      val indexes = Array(0, 0, 0)
      val bwd = MklDnnOps.primitiveCreate2(bwd_pd, inputs, indexes, 3, outputs, 1)

      /* build a simple net */
      stream_bwd.clear()
      stream_bwd.append(bwd)
    }
    val n_bwd = stream_bwd.length
    val memoryPrimitives = Array(src_memory, gradOutput_memory, work_memory, gradInput_memory)
    val buffer = Array(input, gradOutput, workSpace, gradInput)
    MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd, memoryPrimitives, buffer)

    val end1 = (System.nanoTime() - s1)/1e9
    // println(s"lrn dnn backward ${end1}")
    gradInput
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
