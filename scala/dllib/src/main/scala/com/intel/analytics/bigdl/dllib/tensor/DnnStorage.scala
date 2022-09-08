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
package com.intel.analytics.bigdl.dllib.tensor

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.dllib.nn.mkldnn.MemoryOwner
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import org.apache.commons.io.serialization.ValidatingObjectInputStream

import scala.reflect._

/**
 * Represent a native array which is needed by mkl-dnn
 * @param size Storage size
 * @tparam T data type, only support float now
 */
private[tensor] class DnnStorage[T: ClassTag](size: Int) extends Storage[T] {
  private def checkIsInstanceOf(that: Any): Boolean = {
    scala.reflect.classTag[T] == that
  }

  private val bytes = if (checkIsInstanceOf(ClassTag.Float)) {
    DnnStorage.FLOAT_BYTES
  } else if (checkIsInstanceOf(ClassTag.Byte)) {
    DnnStorage.INT8_BYTES
  } else if (checkIsInstanceOf(ClassTag.Int)) {
    DnnStorage.INT_BYTES
  } else {
    Log4Error.invalidInputError(false, s"Unsupported type for storage",
    "DnnStorage only support Float, Byte, Int")
    DnnStorage.INT_BYTES
  }

  private var _isReleased: Boolean = false

  // Hold the address of the native array
  @transient var ptr: Pointer = new Pointer(allocate(size))

  override def length(): Int = size

  override def apply(index: Int): T = {
    Log4Error.invalidOperationError(false, "Not support this operation in DnnStorage")
    0.asInstanceOf[T]
  }

  /**
   * Set the element at position index in the storage. Valid range of index is 1 to length()
   *
   * @param index
   * @param value
   */
  override def update(index: Int, value: T): Unit = {
    Log4Error.invalidOperationError(false, "Not support update in DnnStorage")
  }

  override def copy(source: Storage[T], offset: Int, sourceOffset: Int, length: Int)
  : this.type = {
    source match {
      case s: ArrayStorage[T] =>
        Log4Error.unKnowExceptionError(checkIsInstanceOf(ClassTag.Float),
          s"copy from float storage not supported")
        Memory.CopyArray2Ptr(s.array().asInstanceOf[Array[Float]], sourceOffset,
          ptr.address, offset, length, bytes)
      case s: DnnStorage[T] =>
        Memory.CopyPtr2Ptr(s.ptr.address, sourceOffset, ptr.address, offset, length,
          bytes)
      case _ =>
        Log4Error.invalidOperationError(false, "Only support copy from ArrayStorage or DnnStorage")
    }
    this
  }

  override def fill(value: T, offset: Int, length: Int): DnnStorage.this.type = {
    Log4Error.invalidOperationError(false, "Not support this operation in DnnStorage")
    this
  }

  override def resize(size: Long): DnnStorage.this.type = {
    Log4Error.invalidOperationError(false, "Not support this operation in DnnStorage")
    this
  }

  override def array(): Array[T] = {
    Log4Error.invalidOperationError(false, "Not support this operation in DnnStorage")
    null
  }

  override def set(other: Storage[T]): DnnStorage.this.type = {
    Log4Error.invalidOperationError(false, "Not support this operation in DnnStorage")
    this
  }

  override def iterator: Iterator[T] = {
    Log4Error.invalidOperationError(false, "Not support this operation in DnnStorage")
    null
  }

  /**
   * Release the native array, the storage object is useless
   */
  def release(): Unit = synchronized {
    if (!this.isReleased() && ptr.address != 0L) {
      Memory.AlignedFree(ptr.address)
      DnnStorage.checkAndSet(ptr.address)
      _isReleased = true
      ptr = null
    }
  }

  def isReleased(): Boolean = _isReleased

  private def allocate(capacity: Int): Long = {
    Log4Error.unKnowExceptionError(capacity > 0, s"capacity should be larger than 0")
    val ptr = Memory.AlignedMalloc(capacity * bytes, DnnStorage.CACHE_LINE_SIZE)
    Log4Error.unKnowExceptionError(ptr != 0L, s"allocate native aligned memory failed")
    _isReleased = false
    DnnStorage.add(ptr)
    ptr
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
//    val in1 = new ValidatingObjectInputStream(in)
//    in1.defaultReadObject()
    in.defaultReadObject()
    if (!_isReleased) {
      ptr = new Pointer(allocate(this.size))
      val elements = in.readObject().asInstanceOf[Array[Float]]
      Memory.CopyArray2Ptr(elements, 0, ptr.address, 0, size, DnnStorage.FLOAT_BYTES)
    }
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    if (!_isReleased) {
      val elements = new Array[Float](this.length())
      Memory.CopyPtr2Array(this.ptr.address, 0, elements, 0, size, DnnStorage.FLOAT_BYTES)
      out.writeObject(elements)
    }
  }
}

/**
 * Represent a native point
 * @param address
 */
private[bigdl] class Pointer(val address: Long)

object DnnStorage {
  private[tensor] val CACHE_LINE_SIZE = System.getProperty("bigdl.cache.line", "64").toInt
  private[bigdl] val FLOAT_BYTES: Int = 4
  private[bigdl] val INT8_BYTES: Int = 1
  private[bigdl] val INT_BYTES: Int = 4

  import java.util.concurrent.ConcurrentHashMap
  private val nativeStorages: ConcurrentHashMap[Long, Boolean] = new ConcurrentHashMap()

  def checkAndSet(pointer: Long): Boolean = {
    nativeStorages.replace(pointer, false, true)
  }

  def add(key: Long): Unit = {
    nativeStorages.put(key, false)
  }

  def get(): Map[Long, Boolean] = {
    import scala.collection.JavaConverters._
    nativeStorages.asScala.toMap
  }
}
