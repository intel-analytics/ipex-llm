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
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ReLUDnn[T: ClassTag](ip: Boolean = false, value: Float = 0.0f)(
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


    @transient
    private var reorder_gradOutput_memory: Long = 0L
    var reorder_gradOutput: Long = 0L
    private var inputBuffer : MklDnnTensor[Float] = null
    private var gradOutputBuffer : MklDnnTensor[Float] = null

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
        // todo: output with Dense Tensor
        output = MklDnnTensor[Float](input.size())
        input_format = input.dim() match {
          case 1 => MklDnn.MemoryFormat.nc
          case 2 => MklDnn.MemoryFormat.nc
          case 4 => MklDnn.MemoryFormat.nchw
        }
        if (input.getPrimitiveDesc() != 0L) {
          val input_pd = input.getPrimitiveDesc()
          src_md = MklDnnOps.primitiveDescQueryMemory(input_pd)
          src_memory = MklDnn.PrimitiveCreate0(input_pd)
        } else {
          if (input.dim() == 1) {
            src_md = MklDnnOps.memoryDescInit(1 + input.dim(), Array(1) ++ input.size(),
              MklDnn.DataType.f32, this.input_format)
          } else {
            src_md = MklDnnOps.memoryDescInit(input.dim(), input.size(), MklDnn.DataType.f32,
              this.input_format)
          }
          src_memory = MklDnnOps.createMemoryPrimitive(src_md, engine)
        }

        val relu_desc = MklDnnOps.eltwiseForwardDescInit(MklDnn.PropKind.forward,
          MklDnn.AlgKind.eltwiseRelu, src_md, value, 0)
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

        if (inputBuffer == null && input.getTensorType != MklDnnType) {
          inputBuffer = MklDnnTensor[Float](input.size())
        } else if (inputBuffer != null && inputBuffer.nElement() != input.nElement()) {
          inputBuffer.release()
          inputBuffer = MklDnnTensor[Float](input.size())
        }
      }
      if (System.getProperty("debug") == "1") {
        println("relu updateoutput start " + this.getName())
      }

      if (input.getTensorType != MklDnnType) {
        MklDnnTensor.syncFromHeap(inputBuffer, input.storage().array(), input.storageOffset() - 1)
      } else {
        inputBuffer = input.asInstanceOf[MklDnnTensor[Float]]
      }
      val memoryPrimitives = Array(src_memory, dst_memory)
      val buffer = Array(inputBuffer, output)
      val stream_fwd = Array(relu_fwd)
      val n_fwd = stream_fwd.length
      MklDnnOps.streamSubmit(stream, n_fwd, stream_fwd, n_fwd, memoryPrimitives, buffer)

      val end1 = (System.nanoTime() - s1)/1e6
      if (System.getProperty("debug") == "2") {
        println(s"relu dnn ${this.getName()} forward ${end1}")
      }

      output.layer_name = this.getName()
      output
    }

    override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
      val s1 = System.nanoTime()
      if (update_primitive) {
        var gradOutput_md : Long = 0L
        if (gradOutput.getPrimitiveDesc() != 0L) {
          val gradOutput_pd = gradOutput.getPrimitiveDesc()
          // gradOutput_md = MklDnnOps.primitiveDescQueryMemory(gradOutput_pd)
          gradOutput_memory = MklDnn.PrimitiveCreate0(gradOutput_pd)

          if (input.getFormat() == -1) {
            // gradOutput_md = MklDnnOps.primitiveDescQueryMemory(gradOutput_pd)
            gradOutput_md = MklDnnOps.memoryDescInit(gradOutput.dim(), gradOutput.size(),
              MklDnn.DataType.f32, this.input_format)
          } else {
            gradOutput_md = MklDnnOps.memoryDescInit(gradOutput.dim(), gradOutput.size(),
              MklDnn.DataType.f32, input.getFormat())
          }
        } else {
          val gradOutputFormat =
            if ((input.getFormat() != -1) && (input.getFormat() != this.input_format)) {
            input.getFormat()
          } else {
            this.input_format
          }
          if (gradOutput.dim() == 1) {
            gradOutput_md = MklDnn.MemoryDescInit(gradOutput.dim() + 1,
              Array(1) ++ gradOutput.size(), MklDnn.DataType.f32, gradOutputFormat)
          } else {
            gradOutput_md = MklDnn.MemoryDescInit(gradOutput.dim(), gradOutput.size(),
              MklDnn.DataType.f32, gradOutputFormat)
          }
          gradOutput_memory = MklDnnOps.initDataMemory(gradOutput.dim(), gradOutput.size(),
            this.input_format, MklDnn.DataType.f32, engine)
          // gradOutput_memory = MklDnnOps.createMemoryPrimitive(gradOutput_md, engine)
        }

        /* create backward relu descriptor */
        val bwd_desc = MklDnnOps.eltwiseBackwardDescInit(MklDnn.AlgKind.eltwiseRelu, gradOutput_md,
          src_md, value, 0)
        val bwd_pd = MklDnnOps.primitiveDescCreate(bwd_desc, engine, relu_fwd_pd)

        /* create memory primities for relu diff src */
        // todo: output with Dense Tensor
        gradInput = MklDnnTensor[Float](input.size())
        val gradInput_pd = MklDnnOps.primitiveDescQueryPd(bwd_pd, MklDnn.Query.diff_src_pd, 0)
        gradInput_memory = MklDnn.PrimitiveCreate0(gradInput_pd)
        gradInput.setPrimitiveDesc(gradInput_pd)


        /* create reorder primitives between user gradOutput and pooling gradOutput */
        val res = MklDnnOps.reorderToInternal(gradOutput_memory, bwd_pd, MklDnn.Query.diff_dst_pd,
          gradOutputBuffer, gradOutput.size())
        reorder_gradOutput = res._1
        reorder_gradOutput_memory = res._2

        val internal_gradOutput_memory = if (reorder_gradOutput_memory == 0L) {
          gradOutput_memory
        } else {
          println(s"relu updateGradInput reorder ${this.getName()}")
          reorder_gradOutput_memory
        }

        val relu_inputs = Array(src_memory, internal_gradOutput_memory)
        val indexes = Array(0)
        val relu_outputs = Array(gradInput_memory)

        relu_bwd = MklDnnOps.primitiveCreate2(bwd_pd, relu_inputs, indexes, relu_inputs.length,
          relu_outputs, relu_outputs.length)

        if ((gradOutputBuffer == null && gradOutput.getTensorType != MklDnnType)
          || (reorder_gradOutput_memory != 0L && gradOutputBuffer == null)) {
          gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
        } else if (gradOutputBuffer != null && gradOutputBuffer.nElement() != gradOutput.nElement())
        {
          gradOutputBuffer.release()
          gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
        }
      }

      if (System.getProperty("debug") == "1") {
        println("relu backward " + this.getName())
      }

      val stream_bwd = if (reorder_gradOutput_memory == 0L) {
        Array(relu_bwd)
      } else {
        Array(reorder_gradOutput, relu_bwd)
      }
      val n_bwd = stream_bwd.length
      val (memoryPrimitives, buffer) =
        if (reorder_gradOutput_memory == 0L && gradOutput.getTensorType != MklDnnType) {
          // sync here
          MklDnnTensor.syncFromHeap(
            gradOutputBuffer, gradOutput.storage().array(), gradOutput.storageOffset() - 1)
          (Array(src_memory, gradOutput_memory, gradInput_memory),
            Array(inputBuffer, gradOutputBuffer, gradInput))
        } else {
          (Array(src_memory, gradInput_memory, gradOutput_memory, reorder_gradOutput_memory),
          Array(inputBuffer, gradInput, gradOutput, gradOutputBuffer))
        }

      MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd, n_bwd, memoryPrimitives, buffer)

      val end1 = (System.nanoTime() - s1)/1e6
      if (System.getProperty("debug") == "2") {
        println(s"relu dnn ${this.getName()} backward ${end1}")
      }
      gradInput.layer_name = this.getName()
      gradInput
    }
  }

object ReLUDnn {
  def apply[T: ClassTag](ip: Boolean = false, value: Float = 0.0f)
    (implicit ev: TensorNumeric[T]): ReLUDnn[T] = {
    new ReLUDnn[T](ip, value)
  }
}
