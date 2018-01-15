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
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class MemoryPrimitive[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Serializable {
  class TensorWithPrimitive extends Serializable {
    @transient var handle: Long = 0L
    @transient var desc: Long = 0L
    @transient var primitive: Long = 0L

    val tensor: Tensor[T] = Tensor[T]()
  }

  val user: TensorWithPrimitive = new TensorWithPrimitive
  val internal: TensorWithPrimitive = new TensorWithPrimitive
  @transient var reorder: Long = 0L // reorder operation

  // TODO maybe it's a big tensor, which has get handle from other layers.
  private def setHandle(tensorWithPrimitive: TensorWithPrimitive): Unit = {
    val tensor = tensorWithPrimitive.tensor
    val primitive = tensorWithPrimitive.primitive

    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    val offset = tensor.storageOffset() - 1

    require(tensorWithPrimitive.handle == 0L, s"You should release this handle first")
    tensorWithPrimitive.handle = MklDnn.MemorySetDataHandle(primitive, data, offset)
  }

  private def releaseHandle(tensorWithPrimitive: TensorWithPrimitive): Unit = {
    val tensor = tensorWithPrimitive.tensor
    val handle = tensorWithPrimitive.handle
    val data = tensor.storage().array().asInstanceOf[Array[Float]]
    MklDnn.MemoryReleaseDataHandle(data, handle)
    tensorWithPrimitive.handle = 0L // reset it to 0
  }

  def setHandle(tensor: Tensor[T]): Unit = {
    require(ev.getType() == FloatType, s"only support float tensor currently")

    if (internal.primitive != 0L && !internal.tensor.isEmpty) {
      setHandle(internal)
    }

    // Anyway, we should set handle of user tensor. If there's no internal tensor,
    // we set it for layer, otherwise the internal tensor for reorder and
    // user tensor for layer.
    user.tensor.set(tensor)
    setHandle(user)
  }

  def releaseHandle(): Unit = {
    if (internal.primitive != 0L && !internal.tensor.isEmpty) {
      releaseHandle(internal)
    }
    releaseHandle(user)
  }

  def workPrim(): Long = {
    if (internal.primitive != 0L && !internal.tensor.isEmpty) {
      internal.primitive
    } else {
      user.primitive
    }
  }

  def initUser(tensor: Tensor[T], dataType: Int, format: Int, engine: Long): Unit = {
    if (tensor.getPrimitiveDesc() != 0L) { // if the tensor comes from mkldnn layer
      val primDesc = tensor.getPrimitiveDesc()
      user.primitive = MklDnn.PrimitiveCreate0(primDesc)
      user.desc = MklDnnOps.primitiveDescQueryMemory(primDesc)
    } else {
      val (dim, size) = if (tensor.dim() == 1 && (format == MklDnn.MemoryFormat.nc ||
        format == MklDnn.MemoryFormat.oi)) {
        (2, Array(1) ++ tensor.size())
      } else {
        (tensor.dim(), tensor.size())
      }

      val desc = MklDnn.MemoryDescInit(dim, size, dataType, format)
      val primDesc = MklDnn.MemoryPrimitiveDescCreate(desc, engine)

      user.primitive = MklDnn.PrimitiveCreate0(primDesc)
      user.desc = desc

      MklDnn.PrimitiveDescDestroy(primDesc)
    }
    user.tensor.set(tensor)
  }

  def initUser(tensor: Tensor[T], layerPrimDesc: Long, queryType: Int, index: Int): Unit = {
    val primDesc = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    user.desc = MklDnnOps.primitiveDescQueryMemory(primDesc)
    user.primitive = MklDnn.PrimitiveCreate0(primDesc)
    user.tensor.set(tensor)

    tensor.setPrimitiveDesc(primDesc)
  }

  def initInternal(layerPrimDesc: Long, queryType: Int): Unit = {
    val primDescFromLayer = MklDnnOps.primitiveDescQueryPd(layerPrimDesc, queryType, 0)
    val res = MklDnnOps.prepareReorder(user.primitive, primDescFromLayer, true)

    if (res._1 != 0L) {
      internal.tensor.resize(user.tensor.size())
      internal.tensor.setPrimitiveDesc(primDescFromLayer)
      internal.primitive = res._2
      reorder = res._1
    }
  }

  def reorderToUser(user_md: Long, internal_pd: Long):
  (Long, Long) = {
    if (internal_pd != 0L) {
      MklDnnOps.prepareReorder(user_md, internal_pd, false)
    } else {
      (0L, 0L)
    }
  }
}
