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
import com.intel.analytics.bigdl.tensor.DnnStorage

sealed trait MemoryData extends Serializable {
  def shape: Array[Int]
  def layout: Int
  def dataType: Int
  var heapFormat : Int = -1

  private var _mask: Int = -1
  private var _scales: Array[Float] = Array.emptyFloatArray

  def mask: Int = _mask
  def setMask(s: Int): Unit = _mask = s
  def scales: Array[Float] = _scales
  def setScales(f: Array[Float]): Unit = _scales = f

  def setHeapFormat(f: Int): this.type = {
    heapFormat = f
    this
  }
  def getHeapShape(): Array[Int] = {
    if (layout == Memory.Format.nhwc) { // native shape is nchw
      Array(shape(0), shape(2), shape(3), shape(1))
    } else shape
  }

  def cloneFormat(): MemoryData

  private val UNDEFINED: Long = -1
  private val ERROR: Long = 0

  @transient private var primitive: Long = UNDEFINED
  @transient private var primitiveDesc: Long = UNDEFINED
  @transient private var description: Long = UNDEFINED

  def getMemoryDescription()(implicit owner: MemoryOwner): Long = {
    if (description == UNDEFINED || description == ERROR) {
      checkConsistency(shape, layout)
      description = MklDnnMemory.MemoryDescInit(shape.length, shape, dataType, layout)
    }
    description
  }

  def getPrimitiveDescription(runtime: MklDnnRuntime)(implicit owner: MemoryOwner): Long = {
    require(runtime != null, s"Have you initialized the MklDnnRuntime?")
    if (primitiveDesc == UNDEFINED || primitiveDesc == ERROR) {
      primitiveDesc =
        MklDnnMemory.MemoryPrimitiveDescCreate(getMemoryDescription(), runtime.engine)
    }
    primitiveDesc
  }

  def getPrimitive(runtime: MklDnnRuntime)(implicit owner: MemoryOwner): Long = {
    require(runtime != null, s"Have you initialized the MklDnnRuntime?")
    if (primitive == UNDEFINED || primitive == ERROR) {
      primitive =
        MklDnnMemory.PrimitiveCreate0(getPrimitiveDescription(runtime))
    }
    primitive
  }

  def setPrimitiveDescription(desc: Long): Unit = {
    primitiveDesc = desc
  }

  def setMemoryDescription(desc: Long): Unit = {
    description = desc
  }

  def getRealSize: Long = {
    require(primitiveDesc != UNDEFINED && primitiveDesc != ERROR)
    MklDnn.PrimitiveDescGetSize(primitiveDesc) / getDataTypeBytes
  }

  def getPaddingShape: Array[Int] = {
    require(description != UNDEFINED && description != ERROR)
    Memory.GetPaddingShape(description)
  }

  private def getDataTypeBytes: Int = {
    dataType match {
      case DataType.F32 => DnnStorage.FLOAT_BYTES
      case DataType.S32 => DnnStorage.INT_BYTES
      case DataType.S8 => DnnStorage.INT8_BYTES
      case DataType.U8 => DnnStorage.INT8_BYTES
      case _ => throw new UnsupportedOperationException(s"unsupported data type")
    }
  }

  private def checkConsistency(shape: Array[Int], layout: Int): Unit = {
    val isConsistency = Memory.Format.any == layout || (shape.length match {
      case 1 => layout == Memory.Format.x
      case 2 => layout == Memory.Format.nc || layout == Memory.Format.io ||
        layout == Memory.Format.oi
      case 3 | 4 | 5 => layout != Memory.Format.nc || layout != Memory.Format.x
      case _ => false
    })

    require(isConsistency,
      s"the shape([${shape.mkString(",")}]) of tensor is different from layout(${layout})")
  }
}

case class HeapData(private var _shape: Array[Int], private var _layout: Int,
  private var _dataType: Int = DataType.F32) extends MemoryData {

  override def dataType: Int = _dataType

//  override def setDataType(dataType: Int): Unit = _dataType = dataType

//  override def setShape(shape: Array[Int]): Unit = _shape = shape.clone()
//
//  override def setLayout(layout: Int): Unit = _layout = layout

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.layout
    var d = 0
    while (d < this.shape.length) {
      hash = hash * seed + this.shape(d)
      d += 1
    }

    hash = hash * seed + this.dataType

    hash
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[HeapData]) {
      return false
    }
    val other = obj.asInstanceOf[HeapData]
    if (this.eq(other)) {
      return true
    }
    if (this.layout != other.layout) {
      return false
    }
    if (this.shape == null && other.shape == null) {
      return true
    }
    if (this.shape != null && other.shape != null) {
      if (this.shape.length != other.shape.length) return false
      var i = 0
      while(i < this.shape.length) {
        if (this.shape(i) != other.shape(i)) return false
        i += 1
      }
      return true
    } else {
      return false
    }
  }

  override def toString: String = {
    s"HeapData([${shape.mkString("x")}], ${layout})"
  }

  override def cloneFormat(): MemoryData = new HeapData(_shape, _layout, _dataType)

  def toNative(): NativeData = {
    NativeData(shape, layout)
  }
}

case class NativeData(private var _shape: Array[Int], private var _layout: Int,
  private var _dataType: Int = DataType.F32) extends MemoryData {

  override def shape: Array[Int] = _shape.clone()

  override def layout: Int = _layout

  override def hashCode(): Int = {
    val seed = 41
    var hash = 1
    hash = hash * seed + this.layout
    var d = 0
    while (d < this.shape.length) {
      hash = hash * seed + this.shape(d)
      d += 1
    }

    hash = hash * seed + this.dataType

    hash
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[NativeData]) {
      return false
    }
    val other = obj.asInstanceOf[NativeData]
    if (this.eq(other)) {
      return true
    }
    if (this.layout != other.layout) {
      return false
    }
    if (this.shape == null && other.shape == null) {
      return true
    }
    if (this.shape != null && other.shape != null) {
      if (this.shape.length != other.shape.length) return false
      var i = 0
      while(i < this.shape.length) {
        if (this.shape(i) != other.shape(i)) return false
        i += 1
      }
      return true
    } else {
      return false
    }
  }

  override def toString: String = {
    s"NativeData([${shape.mkString("x")}], ${layout}, ${dataType}, ${mask}, ${scales})"
  }

  override def cloneFormat(): MemoryData = new NativeData(_shape, _layout, _dataType)

  override def dataType: Int = _dataType
}

private[mkldnn] object MemoryData {

  def primitiveOutput(pd: Long): NativeData = {
    operationWant(pd, Query.DstPd, 0)
  }

  def operationWant(primDesc: Long, queryType: Int, index: Int = 0): NativeData = {
    val memoryPrimDesc = MklDnn.PrimitiveDescQueryPd(primDesc, queryType, index)
    val memoryDesc = MklDnn.PrimitiveDescQueryMemory(memoryPrimDesc)
    val shape = Memory.GetShape(memoryDesc)
    val paddingShape = Memory.GetPaddingShape(memoryDesc)
    val layout = Memory.GetLayout(memoryDesc)
    val dataType = Memory.GetDataType(memoryDesc)

    val memory = NativeData(shape, layout, dataType)
    memory.setMemoryDescription(memoryDesc)
    memory.setPrimitiveDescription(memoryPrimDesc)
    memory
  }
}
