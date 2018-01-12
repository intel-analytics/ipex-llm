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
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ReLUDnn[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[Float] {

    override val isMklDnnModel: Boolean = true

    @transient
    private var engine: Long = 0L
    @transient
    private var stream: Long = 0L
    @transient
    private var src_md: Long = 0L
    @transient
    private var src_memory: Long = 0L
    @transient
    private var dst_memory: Long = 0L
    @transient
    private var gradInput_memory: Long = 0L
    @transient
    private var gradOutput_memory: Long = 0L
    @transient
    private var relu_fwd_pd: Long = 0L
    @transient
    private var relu_fwd: Long = 0L
    @transient
    private var relu_bwd: Long = 0L
    @transient
    private var inputElement : Int = 0
    @transient
    private var update_primitive: Boolean = true

    // for relu, just keep internal format same with input format
    private var input_format = MklDnn.MemoryFormat.nchw


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
        output.resizeAs(input)
        input_format = input.dim() match {
          case 1 => MklDnn.MemoryFormat.x
          case 2 => MklDnn.MemoryFormat.nc
          case 4 => MklDnn.MemoryFormat.nchw
        }

        if (input.getPrimitiveDesc() != 0L) {
          val input_pd = input.getPrimitiveDesc()
          src_md = MklDnnOps.primitiveDescQueryMemory(input_pd)
          src_memory = MklDnn.PrimitiveCreate0(input_pd)
        } else {
          src_md = MklDnnOps.memoryDescInit(input.dim(), input.size(), MklDnn.DataType.f32, this.input_format)
          src_memory = MklDnnOps.createMemoryPrimitive(src_md, engine)
        }

        val relu_desc = MklDnnOps.eltwiseForwardDescInit(MklDnn.PropKind.forward,
          MklDnn.AlgKind.eltwiseRelu, src_md, 0, 0)
        relu_fwd_pd = MklDnnOps.primitiveDescCreate(relu_desc, engine, 0L)

        /* create relu dst memory primitive */
        val dst_pd = MklDnnOps.primitiveDescQueryPd(relu_fwd_pd, MklDnn.Query.dst_pd, 0)
        dst_memory = MklDnn.PrimitiveCreate0(dst_pd)
        output.setPrimitiveDesc(dst_pd)

        /* finally create a relu primitive */
        val relu_inputs = Array(src_memory)
        val indexes = Array(0)
        val relu_outputs = Array(dst_memory)
        relu_fwd = MklDnnOps.primitiveCreate2(relu_fwd_pd, relu_inputs, indexes, relu_inputs.length,
          relu_outputs, relu_outputs.length)
      }

      val memoryPrimitives = Array(src_memory, dst_memory)
      val buffer = Array(input, output)
      val stream_fwd = Array(relu_fwd)
      val n_fwd = stream_fwd.length
      MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd, n_fwd, memoryPrimitives, buffer)

      val end1 = (System.nanoTime() - s1)/1e9
      // println(s"relu dnn forward ${end1}")

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
          gradOutput_md = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(), MklDnn.DataType.f32, this.input_format)
          gradOutput_memory = MklDnnOps.createMemoryPrimitive(gradOutput_md, engine)
        }

        /* create backward relu descriptor */
        val bwd_desc = MklDnnOps.eltwiseBackwardDescInit(MklDnn.AlgKind.eltwiseRelu, gradOutput_md, src_md, 0, 0)
        val bwd_pd = MklDnnOps.primitiveDescCreate(bwd_desc, engine, relu_fwd_pd)

        /* create memory primities for relu diff src */
        gradInput.resizeAs(input)
        val gradInput_pd = MklDnnOps.primitiveDescQueryPd(bwd_pd, MklDnn.Query.diff_src_pd, 0)
        gradInput_memory = MklDnn.PrimitiveCreate0(gradInput_pd)
        gradInput.setPrimitiveDesc(gradInput_pd)

        val relu_inputs = Array(src_memory, gradOutput_memory)
        val indexes = Array(0)
        val relu_outputs = Array(gradInput_memory)

        relu_bwd = MklDnnOps.primitiveCreate2(bwd_pd, relu_inputs, indexes, relu_inputs.length,
          relu_outputs, relu_outputs.length)
      }

      val memoryPrimitives = Array(src_memory, gradInput_memory, gradOutput_memory)
      val buffer = Array(input, gradInput, gradOutput)
      val stream_bwd = Array(relu_bwd)
      val n_bwd = stream_bwd.length
      MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd, n_bwd, memoryPrimitives, buffer)

      val end1 = (System.nanoTime() - s1)/1e9
      // println(s"relu dnn backward ${end1}")
      gradInput
    }
  }

object ReLUDnn {
  def apply[T: ClassTag](ip: Boolean = false)(implicit ev: TensorNumeric[T]): ReLUDnn[T] = {
    new ReLUDnn[T](ip)
  }
}
