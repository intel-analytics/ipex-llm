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
import com.intel.analytics.bigdl.mkl.MklDnn.{EngineType, StreamType}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SoftMax[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val nnSoftMax = nn.SoftMax()
  var hasForwarded = false

  // TODO should be refactored to on module
  val engine: Long = MklDnn.EngineCreate(EngineType.cpu, 0)

  var forwardStream = 0L

  var forwardPrim = 0L
  @transient var userSrcMemoryPrim = 0L
  @transient var userDstMemoryPrim = 0L

  private def initDataMemory(dim: Int, dims: Array[Int], format: Int,
    dataType: Int, engine: Long, tensor: Tensor[T]): Long = {
    val primMd = MklDnn.MemoryDescInit(dim, dims, dataType, format)
    val userPd = MklDnn.MemoryPrimitiveDescCreate(primMd, engine)
    val memory = MklDnn.PrimitiveCreate0(userPd)

    MklDnn.PrimitiveDescDestroy(userPd)
    memory
  }

  private def setHandle(tensor: Tensor[T], primitive: Long): Unit = {
    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    val offset = tensor.storageOffset() - 1
    MklDnn.MemorySetDataHandle(primitive, data, offset)
  }

  private def releaseHandles(input: Tensor[T], ptr: Long): Unit = {
    MklDnn.MemoryReleaseDataHandle(
      input.storage().array().asInstanceOf[Array[Float]], ptr)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    if (forwardPrim == 0L) {
      val (format, ndim, dims, axis) = input.dim() match {
        case 1 => (MklDnn.MemoryFormat.x, 1, input.size(), 0)
        case 2 => (MklDnn.MemoryFormat.nc, 2, input.size(), 1)
        case 3 => (MklDnn.MemoryFormat.nchw, 4, Array(1) ++ input.size(), 1)
        case 4 => (MklDnn.MemoryFormat.nchw, 4, input.size(), 1)
        case _ => throw new UnsupportedOperationException(
          s"1 <= input.nDimension() && input.nDimension() <= 4, 1D, 2D, 3D or 4D tensor expected " +
            s"input dimension ${input.nDimension()}")
      }

      val srcMemDesc = MklDnn.MemoryDescInit(ndim, dims, MklDnn.DataType.f32, format)

      // TODO the axis should depend on the input dimension
      // it's always the first dim. Is it correct?
      val opDesc = MklDnn.SoftMaxForwardDescInit(MklDnn.PropKind.forwardInference,
        srcMemDesc, axis)
      val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

      userSrcMemoryPrim = initDataMemory(ndim, dims, format, MklDnn.DataType.f32, engine, input)
      userDstMemoryPrim = initDataMemory(ndim, dims, format, MklDnn.DataType.f32, engine, output)

      val srcs = Array(userSrcMemoryPrim)
      val indexes = Array(0)
      val dsts = Array(userDstMemoryPrim)

      forwardPrim = MklDnn.PrimitiveCreate2(opPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length)
    }

    if (forwardStream == 0L) {
      forwardStream = MklDnn.StreamCreate(StreamType.eager)
    }

    setHandle(input, userSrcMemoryPrim)
    setHandle(output, userDstMemoryPrim)

    MklDnn.StreamSubmit(forwardStream, 1, Array(forwardPrim))

    releaseHandles(input, userSrcMemoryPrim)
    releaseHandles(output, userDstMemoryPrim)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!hasForwarded) {
      nnSoftMax.forward(input)
      hasForwarded = true
    }

    gradInput = nnSoftMax.backward(input, gradOutput)
    gradInput
  }
}

object SoftMax {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T]()
  }
}
