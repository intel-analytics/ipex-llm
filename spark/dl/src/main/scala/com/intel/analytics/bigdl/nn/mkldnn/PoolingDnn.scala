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
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, Initializable, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, MklDnnType, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class PoolingDnn[T: ClassTag](
  kW: Int,
  kH: Int,
  dW: Int = 1,
  dH: Int = 1,
  padW: Int = 0,
  padH: Int = 0,
  format: DataFormat = DataFormat.NCHW,
  algKind : Int = AlgKind.PoolingMax)(implicit ev: TensorNumeric[T])
  extends TensorModule[Float] with Initializable {

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
  private var reorder_gradOutput_memory: Long = 0L
  @transient
  private var update_primitive: Boolean = true

  val stream_fwd = new ArrayBuffer[Long]
  val stream_bwd = new ArrayBuffer[Long]
  val stream_acc = new ArrayBuffer[Long]

  private val input_format = Memory.Format.nchw
  private val internal_format = Memory.Format.any
  private val dataType = DataType.F32

  private val strides = Array(dW, dH)
  private val kernel = Array(kH, kW)
  private var paddingTL : Array[Int] = null
  private var paddingBR: Array[Int] = null

  private var workSpace : MklDnnTensor[Float] = null
  private var inputBuffer : MklDnnTensor[Float] = null
  private var gradOutputBuffer : MklDnnTensor[Float] = null

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
      require(input.dim() == 4 && input.isContiguous())
      val (dimh, dimw, dimc) = format.getHWCDims(input.dim())

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
          require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel" +
            "size pad size($padW,$padH) kernel size($kW, $kH)")
          if (ceilMode) {
            Utils.getOutSizeAndPaddingForDNN(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode)
          } else {
            Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode)
          }
        }

      val padTop = sizes(0)
      val padBottom = sizes(1)
      val padLeft = sizes(2)
      val padRight = sizes(3)
      val oHeight = sizes(4)
      val oWidth = sizes(5)

      paddingTL = Array(padTop, padLeft)
      paddingBR = Array(padBottom, padRight)

      val nbatch = input.size(1)
      val input_size = input.size()
      // todo: dst size not correct some times
      val dst_sizes = Array(nbatch, nInputPlane, oHeight, oWidth)
      // todo: output with Dense Tensor
      if (output.getTensorType != MklDnnType) {
        output = MklDnnTensor[Float](dst_sizes)
      } else if (output.nElement() != dst_sizes.product) {
        output.asInstanceOf[MklDnnTensor[Float]].release()
        output = MklDnnTensor[Float](dst_sizes)
      }

      // create memory desc, for input
      if (input.getPrimitiveDesc() != 0L) {
        val input_pd = input.getPrimitiveDesc()
        src_memory = MklDnn.PrimitiveCreate0(input_pd)
        src_md = MklDnnOps.primitiveDescQueryMemory(input_pd)
      } else {
        src_md = MklDnnOps.memoryDescInit(input.dim(), input.size(), dataType, this.input_format)
        src_memory = MklDnnOps.createMemoryPrimitive(src_md, engine)
      }

      // for output
      val dst_md = MklDnnOps.memoryDescInit(output.dim(), dst_sizes, dataType, this.internal_format)

      /* create a convolution */
      val pool_desc = MklDnnOps.poolingForwardDescInit(
                        PropKind.Forward, algKind,
                        src_md, dst_md, strides, kernel, paddingTL, paddingBR,
                        MklDnn.PaddingKind.mkldnnPaddingZero)

      fwd_pd = MklDnnOps.primitiveDescCreate(pool_desc, engine, 0L)

      /* create memory for dst data, we don't need to reorder it to user data */
      dst_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, Query.DstPd, 0)
      dst_memory = MklDnn.PrimitiveCreate0(dst_pd)
      output.setPrimitiveDesc(dst_pd)

      val workdspace_pd = MklDnnOps.primitiveDescQueryPd(fwd_pd, Query.WorkspacePd, 0)
      work_memory = MklDnn.PrimitiveCreate0(workdspace_pd)
      val workdspace_size = MklDnn.PrimitiveDescGetSize(workdspace_pd)
      if (workSpace == null) workSpace = MklDnnTensor[Float](Array(workdspace_size.toInt))


      val inputs = Array(src_memory)
      val outputs = Array(dst_memory, work_memory)
      val indexes = Array(0)

      val fwd = MklDnnOps.primitiveCreate2(fwd_pd, inputs, indexes, 1, outputs, 2)

      /* build a simple net */
      stream_fwd.clear()
      stream_fwd.append(fwd)

      if (inputBuffer == null && input.getTensorType != MklDnnType) {
        inputBuffer = MklDnnTensor[Float](input.size())
      } else if (inputBuffer != null && inputBuffer.nElement() != input.nElement()) {
        inputBuffer.release()
        inputBuffer = MklDnnTensor[Float](input.size())
      }
    }
    if (System.getProperty("debug") == "1") {
      println("pooling updateoutput start " + this.getName())
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
      DnnTools.debugFwInfo(this.getName(), end1, input.getFormat(), output.getFormat())
    }
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
        gradOutput_md = MklDnn.PrimitiveDescQueryMemory(dst_pd)
        gradOutput_memory = MklDnnOps.initDataMemory(gradOutput.dim(), gradOutput.size(),
          this.input_format, dataType, engine)
      }

      // for gradInput
      // todo: output with Dense Tensor
      gradInput = MklDnnTensor[Float](input.size())
      val gradInput_md = MklDnnOps.memoryDescInit(gradInput.dim(), gradInput.size(),
        dataType, this.internal_format)

      /* create backward descriptor */
      val bwd_desc = MklDnnOps.poolingBackwardDescInit(algKind, gradInput_md, gradOutput_md,
                    strides, kernel, paddingTL, paddingBR, MklDnn.PaddingKind.mkldnnPaddingZero)
      val bwd_pd = MklDnnOps.primitiveDescCreate(bwd_desc, engine, fwd_pd)


      /* create memory primities for relu diff src */
      val gradInput_pd = MklDnnOps.primitiveDescQueryPd(bwd_pd, Query.DiffSrcPd, 0)
      gradInput_memory = MklDnn.PrimitiveCreate0(gradInput_pd)
      gradInput.setPrimitiveDesc(gradInput_pd)

      /* create reorder primitives between user gradOutput and pooling gradOutput */
      var reorder_gradOutput: Long = 0L
      val res = MklDnnOps.reorderToInternal(gradOutput_memory, bwd_pd, Query.DiffDstPd,
        gradOutputBuffer, gradOutput.size())
      reorder_gradOutput = res._1
      reorder_gradOutput_memory = res._2

      val internal_gradOutput_memory = if (reorder_gradOutput_memory == 0L) {
        gradOutput_memory
      } else {
        println("pool updateGradInput reorder")
        reorder_gradOutput_memory
      }

      val inputs = Array(internal_gradOutput_memory, work_memory)
      val outputs = Array(gradInput_memory)
      val indexes = Array(0, 0)
      val bwd = MklDnnOps.primitiveCreate2(bwd_pd, inputs, indexes, 2, outputs, 1)

      /* build a simple net */
      stream_bwd.clear()
      if (reorder_gradOutput_memory != 0L) stream_bwd.append(reorder_gradOutput)
      stream_bwd.append(bwd)

      if ((gradOutputBuffer == null && gradOutput.getTensorType != MklDnnType)
        || (reorder_gradOutput_memory != 0L && gradOutputBuffer == null)) {
        gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
      } else if (gradOutputBuffer != null && gradOutputBuffer.nElement() != gradOutput.nElement()) {
        gradOutputBuffer.release()
        gradOutputBuffer = MklDnnTensor[Float](gradOutput.size())
      }
    }
    val n_bwd = stream_bwd.length
    val (memoryPrimitives, buffer) =
      if (reorder_gradOutput_memory == 0L && gradOutput.getTensorType != MklDnnType) {
      // sync here
      MklDnnTensor.syncFromHeap(
        gradOutputBuffer, gradOutput.storage().array(), gradOutput.storageOffset() - 1)
      (Array(gradOutput_memory, work_memory, gradInput_memory),
      Array(gradOutputBuffer, workSpace, gradInput))
    } else {
      (Array(gradOutput_memory, reorder_gradOutput_memory, work_memory, gradInput_memory),
        Array(gradOutput, gradOutputBuffer, workSpace, gradInput))
    }

    MklDnnOps.streamSubmit(stream, n_bwd, stream_bwd.toArray, n_bwd, memoryPrimitives, buffer)

    val end1 = (System.nanoTime() - s1)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, gradOutput.getFormat(), gradInput.getFormat())
    }
    gradInput
  }

  override def clearState() : this.type = {
    super.clearState()
    if (gradOutputBuffer != null) {
      gradOutputBuffer.release()
      gradOutputBuffer.set()
    }
    if (workSpace != null) {
      workSpace.release()
      workSpace.set()
    }
    if (inputBuffer != null) {
      inputBuffer.release()
      inputBuffer.set()
    }
    this
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
    new PoolingDnn[T](kW, kH, dW, dH, padW, padH, format, AlgKind.PoolingMax)
  }
}
