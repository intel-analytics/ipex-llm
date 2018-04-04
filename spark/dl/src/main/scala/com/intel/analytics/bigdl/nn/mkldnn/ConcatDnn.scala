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

import breeze.linalg.dim
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{Container, DynamicContainer, JoinTable}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{MklDnnTensor, MklDnnType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * Concat concatenates the output of one layer of "parallel"
 * modules along the provided {@code dimension}: they take the
 * same inputs, and their output is concatenated.
 *                 +-----------+
 *            +---->  module1  -----+
 *            |    |           |    |
 * input -----+---->  module2  -----+----> output
 *            |    |           |    |
 *            +---->  module3  -----+
 *                 +-----------+
 *
 * @param dimension dimension
 */
@SerialVersionUID(- 5218461876031660707L)
class ConcatDnn(val dimension: Int) extends DynamicContainer[Tensor[Float], Tensor[Float], Float] {
  private var size: Array[Int] = null
  @transient
  private var resultsMPD: Array[Long] = null
  @transient
  private var resultsPrimitive: Array[Long] = null
  @transient
  private var gradouts: Array[Tensor[Float]] = null
  @transient
  private var outBuffers: Array[MklDnnTensor[Float]] = null
  @transient
  private var results: Array[Future[Unit]] = null

  protected var forwardTimeOverhead = 0L

  private var user_format = 0

  @transient
  private var inputElement : Int = 0
  @transient
  private var update_primitive: Boolean = true
  @transient
  private var engine: Long = 0L
  @transient
  private var stream: Long = 0L
  @transient
  private var dst_memory: Long = 0L

  val stream_fwd = new ArrayBuffer[Long]
  private var stream_reOrder: Array[Array[Long]] = null
  private var reorder_src_memory: Array[Long] = null
  private var reorder_dst_memory: Array[Long] = null

  val dataType = MklDnn.DataType.f32

  def reorderToUser(input: Tensor[Float], output: Tensor[Float],
                    outputFormat: Int, num: Int): Unit = {
    if (update_primitive) {
      val sizes = input.size()
      val dim = input.dim()
      output.resizeAs(input)

      val src_pd = input.getPrimitiveDesc()
      reorder_dst_memory(num) = MklDnnOps.initDataMemory(dim, sizes, outputFormat, dataType, engine)
      val res = MklDnnOps.prepareReorder(reorder_dst_memory(num), src_pd, false)

      stream_reOrder(num)(0) = res._1
      reorder_src_memory(num) = res._2
    }
    /* build a simple net */
    val memoryPrimitives = Array(reorder_src_memory(num), reorder_dst_memory(num))
    val buffer = Array(input, output)
    MklDnnOps.streamSubmit(stream, 1, stream_reOrder(num), 1, memoryPrimitives, buffer)
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    val s1 = System.nanoTime()
    var allModelsTime : Long = 0L
    if (outBuffers == null) outBuffers = new Array[MklDnnTensor[Float]](modules.length)
    if (resultsMPD == null) resultsMPD = new Array[Long](this.modules.length)
    if (resultsPrimitive == null) resultsPrimitive = new Array[Long](this.modules.length)
    if (stream_reOrder == null) {
      stream_reOrder = new Array[Array[Long]](this.modules.length)
      for (i <- 0 until this.modules.length) {
        stream_reOrder(i) = new Array[Long](1)
      }
    }
    if (reorder_dst_memory == null) { reorder_dst_memory = new Array[Long](modules.length) }
    if (reorder_src_memory == null) { reorder_src_memory = new Array[Long](modules.length) }

    if (engine == 0L) engine = this.getDnnEngine(0)
    if (stream == 0L) stream = this.getStream()

    if (inputElement != input.nElement()) {
      update_primitive = true
      inputElement = input.nElement()
    } else {
      update_primitive = false
    }

    var default_format = 0
    var i = 0
    while (i < this.modules.length) {
      val s1 = System.nanoTime()
      val currentOut = this.modules(i).
                        updateOutput(input.asInstanceOf[Activity]).asInstanceOf[Tensor[Float]]
      allModelsTime += System.nanoTime() - s1
      if (i == 0) {
        default_format = currentOut.dim() match {
          case 1 => MklDnn.MemoryFormat.x
          case 2 => MklDnn.MemoryFormat.nc
          case 4 => MklDnn.MemoryFormat.nchw
        }
        user_format = currentOut.getFormat() match {
          case -1 => default_format
          case _ => currentOut.getFormat()
        }
      }

      val format = currentOut.getFormat() match {
        case -1 => default_format
        case _ => currentOut.getFormat()
      }

      if (update_primitive) {
        val md = MklDnnOps.memoryDescInit(currentOut.dim(), currentOut.size(), dataType, user_format)
        resultsMPD(i) = MklDnnOps.memoryPrimitiveDescCreate(md, engine)
        resultsPrimitive(i) = MklDnn.PrimitiveCreate0(resultsMPD(i))
      }

      // reorder all output to user format
      if (outBuffers(i) == null) {
        outBuffers(i) = MklDnnTensor[Float](currentOut.size())
      } else if (outBuffers(i) != null && outBuffers(i).nElement() != currentOut.nElement()) {
        outBuffers(i).release()
        outBuffers(i) = MklDnnTensor[Float](currentOut.size())
      }
      if (format != user_format) {
        // if (outBuffers(i) == null) outBuffers(i) = MklDnnTensor[Float](currentOut.size())
        reorderToUser(currentOut, outBuffers(i), user_format, i)
      } else if (currentOut.getTensorType != MklDnnType) {
        // copy results from normal tensor to dnn tensor
        // if (outBuffers(i) == null) outBuffers(i) = MklDnnTensor[Float](currentOut.size())
        MklDnnTensor.syncFromHeap(outBuffers(i),
                                currentOut.storage().array(), currentOut.storageOffset() - 1)
      } else {
        outBuffers(i) = currentOut.asInstanceOf[MklDnnTensor[Float]]
      }

      if (i == 0) {
        this.size = currentOut.size()
      } else {
        require(this.size.length == currentOut.size.length,
          s"${this.modules(i).getName} output size mismatch, expected : ${this.size.length}," +
            s"actual ${currentOut.size.length}")
        var index = 0
        val ssize = this.size.length
        while (index < ssize) {
          if (index != dimension - 1) {
            require(this.size(index) == currentOut.size(index + 1),
              s"${this.modules(i).getName} output size at dimension ${index + 1} mismatch," +
                s"expected ${this.size(index)}, actual : ${currentOut.size(index + 1)}")
          }
          index += 1
        }
        this.size(this.dimension - 1) += currentOut.size(this.dimension)
      }
      i += 1
    }
    if (output.getTensorType != MklDnnType) {
      // todo: output with Dense Tensor
      output = MklDnnTensor[Float](this.size)
    } else if (output.nElement() != this.size.product) {
      println("ConcatDnn: resize output")
      output.asInstanceOf[MklDnnTensor[Float]].release()
      output = MklDnnTensor[Float](this.size)
    }

    if (update_primitive) {
      val dst_md = MklDnnOps.memoryDescInit(output.dim(), output.size(), dataType, user_format)
      val concatDesc = MklDnn.ConcatPrimitiveDescCreate(dst_md,
        modules.length, dimension - 1, resultsMPD)

      // for dst memory
      val dst_mpd = MklDnnOps.primitiveDescQueryPd(concatDesc, MklDnn.Query.dst_pd, 0)
      output.setPrimitiveDesc(dst_mpd)
      dst_memory = MklDnn.PrimitiveCreate0(dst_mpd)

      val inputs = resultsPrimitive // input memory
      val outputs = Array(dst_memory) // dst memory
      val indexes = new Array[Int](this.modules.length)

      val fwd = MklDnnOps.primitiveCreate2(concatDesc, inputs, indexes, modules.length, outputs, 1)
      stream_fwd.clear()
      stream_fwd.append(fwd)
    }

    val length = stream_fwd.length
    // todo: maybe gc
    val memoryPrimitives = resultsPrimitive ++ Array(dst_memory)
    val buffer = outBuffers ++ Array(output)
    MklDnnOps.streamSubmit(stream, length, stream_fwd.toArray, length, memoryPrimitives, buffer)

    val end1 = (System.nanoTime() - s1 - allModelsTime)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugFwInfo(this.getName(), end1, input.getFormat(), output.getFormat())
    }
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    throw new UnsupportedOperationException(s"ConcatDnn: Unimplemented method")
  }

  override def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    var before = System.nanoTime()
    var allModelsTime : Long = 0L
    if (gradouts == null || gradouts.length != this.modules.length) {
      gradouts = new Array[Tensor[Float]](this.modules.length)
    }
    if (results == null || results.length != this.modules.length) {
      results = new Array[Future[Unit]](this.modules.length)
    }

    if (gradInput.getTensorType != MklDnnType) {
      gradInput = MklDnnTensor[Float](input.size())
    }

    var i = 0
    var offset = 1
    while (i < this.modules.length) {
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[Float]]
      val _offset = offset
      val _i = i
      results(i) = Engine.model.invoke( () => {
        val narrowedTensor = gradOutput.narrow(dimension, _offset, currentOutput.size(dimension))
        if (gradOutput.getPrimitiveDesc() != 0L) {
          var pd: Long = 0L
          if (update_primitive) {
            val md = MklDnnOps.memoryDescInit(
              narrowedTensor.dim(), narrowedTensor.size(), dataType, gradOutput.getFormat())
            pd = MklDnnOps.memoryPrimitiveDescCreate(md, engine)
          }
          narrowedTensor.setPrimitiveDesc(pd)
        }
        if(dimension == 2) {
          gradouts(_i) = Tensor[Float]().resizeAs(narrowedTensor)
          if (narrowedTensor.getPrimitiveDesc() != 0L) {
            gradouts(_i).setPrimitiveDesc(narrowedTensor.getPrimitiveDesc())
          }
          var b = 1
          val firstSize = narrowedTensor.size(1)
          while(b <= firstSize) {
            gradouts(_i).select(1, b).copy(narrowedTensor.select(1, b))
            b += 1
          }
        } else {
          gradouts(_i) = narrowedTensor.contiguous()
        }
      })
      i += 1
      offset += currentOutput.size(dimension)
    }
    Engine.model.sync(results)

    var format: Int = 0
    i = 0
    offset = 1
    while (i < this.modules.length) {
      val s1 = System.nanoTime()
      val currentOutput = this.modules(i).output.asInstanceOf[Tensor[Float]]
      val currentGradInput = this.modules(i).backward(input.asInstanceOf[Activity],
                                gradouts(i).asInstanceOf[Activity]).asInstanceOf[Tensor[Float]]

      allModelsTime += System.nanoTime() - s1
      if (currentGradInput != null) {
        if (i == 0) {
          format = currentGradInput.getFormat()
          gradInput.copy(currentGradInput)
        } else {
          // todo: before add, need to transfor to same format, here just require
          gradInput.add(currentGradInput)
        }
      }
      i += 1
      offset += currentOutput.size(dimension)
    }

    val end1 = (System.nanoTime() - before - allModelsTime)/1e6
    if (System.getProperty("debug") == "2") {
      DnnTools.debugBwInfo(this.getName(), end1, format, gradInput.getFormat())
    }
    gradInput
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[ConcatDnn]) {
      return false
    }
    val other = obj.asInstanceOf[ConcatDnn]
    if (this.eq(other)) {
      return true
    }
    if (dimension != other.dimension) {
      return false
    }

    if (this.modules.length != other.modules.length) {
      return false
    }

    val moduleLength = modules.length
    var i = 0
    while (i < moduleLength) {
      if (modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }

  override def hashCode() : Int = {

    val seed = 37
    var hash = super.hashCode()
    var i = 0
    val moduleLength = modules.length
    while (i < moduleLength) {
      hash = hash * seed + modules(i).hashCode()
      i += 1
    }

    hash
  }
}

object ConcatDnn {
  def apply(dimension: Int) : ConcatDnn = {
    new ConcatDnn(dimension)
  }
}
