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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.mkl.MklDnn.{EngineType, StreamType}

import scala.reflect.ClassTag

class ReLU[T: ClassTag](ip: Boolean = false)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  MklDnn.isLoaded

  val engine = MklDnn.EngineCreate(EngineType.cpu, 0)

  def initDataMemory(dim: Int, dims: Array[Int], format: Int,
    dataType: Int, engine: Long, tensor: Tensor[T]): Long = {
    val primMd = MklDnn.MemoryDescInit(dim, dims, dataType, format)
    val userPd = MklDnn.MemoryPrimitiveDescCreate(primMd, engine)
    val memory = MklDnn.PrimitiveCreate0(userPd)

    val req1 = MklDnn.MemoryGetDataHandle(memory)
    require(req1 == 0)

    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    val offset = tensor.storageOffset() - 1
    val req2 = MklDnn.MemorySetDataHandle(memory, data, offset)
    MklDnn.PrimitiveDescDestroy(userPd)
    memory
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val userPrimitiveDesc = initDataMemory(4, input.size(), MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, engine, input)
    val primitiveDesc = MklDnn.MemoryDescInit(4, input.size(), MklDnn.DataType.f32,
      MklDnn.MemoryFormat.nchw)

    val opDesc = MklDnn.EltwiseForwardDescInit(MklDnn.PropKind.forward,
      MklDnn.AlgKind.eltwiseRelu,
      primitiveDesc,
      1.0f,
      0)

    val opPrimDesc = MklDnn.PrimitiveDescCreate(opDesc, engine, 0)

    output.resizeAs(input)
    val dstPrimDesc = initDataMemory(4, input.size(), MklDnn.MemoryFormat.nchw,
      MklDnn.DataType.f32, engine, output)

    val primtive = MklDnn.PrimitiveCreate2(opPrimDesc, Array(userPrimitiveDesc), Array(0), 1,
      Array(dstPrimDesc), 1)

    val stream = MklDnn.StreamCreate(StreamType.eager)
    MklDnn.StreamSubmit(stream, 1, Array(primtive))
    MklDnn.StreamWait(stream, 1)
    println("=" * 80)
    println(input)
    println("-" * 80)
    println(output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }
}

object ReLU {
  def apply[T: ClassTag](ip: Boolean = false)(implicit ev: TensorNumeric[T]): ReLU[T] = {
    new ReLU[T](ip)
  }
}
