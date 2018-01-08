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
import com.intel.analytics.bigdl.nn.{SpatialMaxPooling, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class PoolingDnn[T: ClassTag](
  kW: Int,
  kH: Int,
  dW: Int = 1,
  dH: Int = 1,
  padW: Int = 0,
  padH: Int = 0,
  val format: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T]) extends TensorModule[Float] with Initializable {

  var ceilMode = false
  val indices = Tensor[T]()

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kW, kH, kW, kH, format = DataFormat.NCHW)
  }

  /**
    * set ceil mode
    * @return this
    */
  def ceil(): PoolingDnn[T] = {
    ceilMode = true
    this
  }

  /**
    * set floor mode
    * @return this
    */
  def floor(): PoolingDnn[T] = {
    ceilMode = false
    this
  }

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
  private val input_format = MklDnn.MemoryFormat.nchw
  private val internal_format = this.input_format
  private val dataType = MklDnn.DataType.f32

  private val strides = Array(dW, dH)
  private val kernel = Array(kH, kW)
  private var paddingTB : Array[Int] = null
  private var paddingLR: Array[Int] = null

  private var workSpace = Tensor[Float]()

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    if (inputElement != input.nElement()) {
      update_primitive = true
      inputElement = input.nElement()
    } else {
      update_primitive = false
    }
    if (update_primitive) {
      require(input.dim() == 4 && input.isContiguous())
      val (dimh, dimw, dimc)  = format.getHWCDims(input.dim())

      val nInputPlane = input.size(dimc)
      val inputHeight = input.size(dimh)
      val inputWidth = input.size(dimw)

      val sizes =
        if (padW == -1 && padH == -1) {
          // no ceil/floor mode in SAME padding
          Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW)
        } else {
          require(inputWidth >= kW - padW && inputHeight >= kH - padH,
            "input smaller than kernel size" +
              s"input size(${input.size(dimw)},${input.size(dimh)})" +
              s"kernel size(${kW-padW},${kH-padH})")
          require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel size" +
            s"pad size($padW,$padH)" +
            s"kernel size($kW, $kH)")
          Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode)
        }

      val padTop = sizes(0)
      val padBottom = sizes(1)
      val padLeft = sizes(2)
      val padRight = sizes(3)
      val oHeight = sizes(4)
      val oWidth = sizes(5)

      paddingTB = Array(padTop, padBottom)
      paddingLR = Array(padLeft, padRight)

      val nbatch = input.size(1)
      val input_size = input.size()
      val dst_sizes = Array(nbatch, nInputPlane, oHeight, oWidth)
      output.resize(dst_sizes)

      // create memory desc, for input
      if (input.getPrimitiveDesc() != 0L) {
        val input_pd = input.getPrimitiveDesc()
        src_memory = MklDnn.PrimitiveCreate0(input_pd)
        src_md = MklDnnOps.primitiveDescQueryMemory(input_pd)
      } else {
        src_md = MklDnnOps.memoryDescInit(input.dim(), input.size(), MklDnn.DataType.f32, this.input_format)
        src_memory = MklDnnOps.createMemoryPrimitive(src_md, engine)

      }

      // for output
      val dst_md = MklDnnOps.memoryDescInit(output.dim(), dst_sizes, dataType, this.internal_format)

      /* create a convolution */
      val pool_desc = MklDnnOps.poolingForwardDescInit(
                        MklDnn.PropKind.forward, MklDnn.AlgKind.poolingMax,
                        src_md, dst_md, strides, kernel, paddingLR, paddingLR,
                        MklDnn.PaddingKind.mkldnnPaddingZero)

      fwd_pd = MklDnnOps.primitiveDescCreate(pool_desc, engine, 0L)

      /* create memory for dst data, we don't need to reorder it to user data */
      val dst_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, MklDnn.Query.dst_pd, 0)
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

    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
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

      // for gradInput
      gradInput.resizeAs(input)
      val gradInput_md = MklDnnOps.memoryDescInit(gradInput.dim(), gradInput.size(),
        dataType, this.internal_format)
      gradInput_memory = MklDnnOps.createMemoryPrimitive(gradInput_md, engine)

      /* create backward descriptor */
      val bwd_desc = MklDnnOps.poolingBackwardDescInit(MklDnn.AlgKind.poolingMax, gradInput_md,
                                                  gradOutput_md, strides, kernel, paddingTB,
                                                  paddingLR, MklDnn.PaddingKind.mkldnnPaddingZero)
      val bwd_pd = MklDnnOps.primitiveDescCreate(bwd_desc, engine, fwd_pd)


      val inputs = Array(gradOutput_memory, work_memory)
      val outputs = Array(gradInput_memory)
      val indexes = Array(0, 0)
      val bwd = MklDnnOps.primitiveCreate2(bwd_pd, inputs, indexes, 2, outputs, 1)

      /* build a simple net */
      stream_bwd.clear()
      stream_bwd.append(bwd)
    }
    val n_bwd = stream_bwd.length
    val memoryPrimitives = Array(gradOutput_memory, work_memory, gradInput_memory)
    val buffer = Array(gradOutput, workSpace, gradInput)
    MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd, memoryPrimitives, buffer)

    gradInput
  }
}

object PoolingDnn {
  def apply[@specialized(Float, Double) T: ClassTag](
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    format: DataFormat = DataFormat.NCHW)
  (implicit ev: TensorNumeric[T]): PoolingDnn[T] = {
    new PoolingDnn[T](kW, kH, dW, dH, padW, padH)
  }
}
