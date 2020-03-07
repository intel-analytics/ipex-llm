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

import com.intel.analytics.bigdl.dnnl._
import com.intel.analytics.bigdl.dnnl.DataType
import com.intel.analytics.bigdl.tensor.DnnStorage


sealed trait MemoryData extends Serializable {

  protected val UNDEFINED: Long = -1
  protected val ERROR: Long = 0

  private var _mask: Int = -1
  private var _scales: Array[Float] = Array.emptyFloatArray

//  @transient private var primitive: Long = UNDEFINED
//  @transient private var primitiveDesc: Long = UNDEFINED
//  @transient private var description: Long = UNDEFINED

  // TODO: would like to make it to val
  @transient protected var memoryDescriptor: Long = UNDEFINED
  @transient protected var memoryObject: Long = UNDEFINED

  def shape: Array[Int]
  def layout: Int
  def dataType: Int
  def mask: Int = _mask
  def scales: Array[Float] = _scales
  def setMask(s: Int): Unit = _mask = s
  def setScales(f: Array[Float]): Unit = _scales = f
  def cloneFormat(): MemoryData

  // refactored getMemoryDescriptor to getMemoryDescriptor
  def getMemoryDescriptor()(implicit owner: MemoryOwner): Long = {
    if (memoryDescriptor == UNDEFINED || memoryDescriptor == ERROR) {
      if (layout == Memory.FormatTag.undef) {
        memoryDescriptor = DnnlMemory.MemoryDescInitByStrides(shape.length, shape, dataType)
      } else {
        checkConsistency(shape, layout)
        memoryDescriptor = DnnlMemory.MemoryDescInit(shape.length, shape, dataType, layout)
      }
    }
    memoryDescriptor
  }


  // replace getMemoryObject to getMemoryObject
  def getMemoryObject(runtime: MklDnnRuntime)(implicit owner: MemoryOwner): Long = {
    require(runtime != null, s"Have you initialized the MklDnnRuntime?")
    if (memoryObject == UNDEFINED || memoryObject == ERROR) {
      memoryObject = DNNL.MemoryCreate(getMemoryDescriptor(), runtime.engine)
    }
    memoryObject
  }

  def setMemoryDescriptor(md: Long): Unit = {
    memoryDescriptor = md
  }

  def setMemoryObject(memory: Long): Unit = {
    memoryObject = memory
  }


  def toNative(): NativeData = {
    this match {
      case native: NativeData => native
      case heap: HeapData => heap.toNative()
      case _ => throw new UnsupportedOperationException("Unsupported memory format")
    }
  }

  /**
   * Returns the size (in bytes) that is required for given @p memory_desc
   * size_t DNNL_API dnnl_memory_desc_get_size(
   * const dnnl_memory_desc_t *memory_desc);
   *
   * @return
   */
  def getRealSize: Long = {
    require(memoryDescriptor != UNDEFINED && memoryDescriptor != ERROR)
    require(getDataTypeBytes != 0)
    DNNL.getSize(memoryDescriptor) / getDataTypeBytes
  }


  def getPaddingShape: Array[Int] = {
    require(memoryDescriptor != UNDEFINED && memoryDescriptor != ERROR)
//    Memory.GetPaddingShape(memoryDescriptor)
    Memory.GetPaddingShape(memoryDescriptor).map(_.toInt)
  }

  private def getDataTypeBytes: Int = {
    dataType match {
      case DataType.F32 => DnnStorage.FLOAT_BYTES
      case DataType.S32 => DnnStorage.INT_BYTES
      case DataType.S8 => DnnStorage.INT8_BYTES
      case DataType.U8 => DnnStorage.INT8_BYTES
      case _ => throw new UnsupportedOperationException(s"unsupported data type: " + dataType)
    }
  }

  private def checkConsistency(shape: Array[Int], layout: Int): Unit = {
    val isConsistency = Memory.FormatTag.any == layout || (shape.length match {
      case 1 => layout == Memory.FormatTag.x
      case 2 => layout == Memory.FormatTag.nc || layout == Memory.FormatTag.io ||
        layout == Memory.FormatTag.oi
      case 3 | 4 | 5 => layout != Memory.FormatTag.nc || layout != Memory.FormatTag.x
      case _ => false
    })

    require(isConsistency,
      s"the shape([${shape.mkString(",")}]) of tensor is different from layout(${layout})")
  }
}



case class HeapData(
  private var _shape: Array[Int],
  private var _layout: Int,
  private var _dataType: Int = DataType.F32) extends MemoryData {

  override def dataType: Int = _dataType

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
    if (memoryDescriptor != UNDEFINED && memoryDescriptor != ERROR) {
      hash = hash * seed + getRealSize.toInt
      hash = hash * seed + getPaddingShape.product
    }

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

  override def toNative(): NativeData = {
    NativeData(shape, layout)
  }
}

case class NativeData(
  private var _shape: Array[Int],
  private var _layout: Int,
  private var _dataType: Int = DataType.F32
) extends MemoryData {

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
    if (memoryDescriptor != UNDEFINED && memoryDescriptor != ERROR) {
      hash = hash * seed + getRealSize.toInt
      hash = hash * seed + getPaddingShape.product
    }

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
    }
    // hotfix: because the layout not working now.
    val owner = new MemoryOwner {}

    val descIsEqual = DNNL.MemoryPrimitiveDescEqual(this.getMemoryDescriptor()(owner),
      other.getMemoryDescriptor()(owner))

    if (descIsEqual == 1) {
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
    operationWant(pd, Query.DstMd, 0)
  }

  def operationWant(pd: Long, queryType: Int, index: Int = 0): NativeData = {
      val memoryDescriptor = DNNL.PrimitiveDescQueryMd(pd, queryType, index)
      if (memoryDescriptor == 0L) {
        return null
      }
      val shape = Memory.GetShape(memoryDescriptor).map(_.toInt)
      val paddingShape = Memory.GetPaddingShape(memoryDescriptor)
      val layout = Memory.GetLayout(memoryDescriptor)
      val dataType = Memory.GetDataType(memoryDescriptor)
      val memory = NativeData(shape, layout, dataType)
      memory.setMemoryDescriptor(memoryDescriptor)
      memory
  }

  def cloneFormatWithDesc(data: MemoryData)(implicit owner: MemoryOwner): MemoryData = {
    val format = data.cloneFormat()
    val desc = DNNL.MemoryDescClone(data.getMemoryDescriptor())
    format.setMemoryDescriptor(desc)
    format
  }
}
