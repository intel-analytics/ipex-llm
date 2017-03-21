/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.parameters

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}

import scala.reflect._

class UnCompressedTensor[T: ClassTag](buffer: Array[Byte], bufferOffset: Int, bufferLength: Int)
  extends CompressedTensor[T] {
  import UnCompressedTensor.tLength

  def this(length: Int) {
    this(new Array[Byte](length), 0, length)
  }

  def this(tensor: Tensor[T]) {
    this(tensor.nElement() * UnCompressedTensor.tLength[T]())
    compress(tensor)
  }

  def this(bytes: ByteBuffer) = this(bytes.array(), bytes.position(), bytes.remaining())

  require(bufferLength % tLength[T]() == 0 &&
    bufferOffset + bufferLength <= buffer.length)

  override def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int)
  : this.type = {
    require(src.isContiguous() && offset >= 0 && srcOffset >= 0 &&
      srcOffset + length <= src.nElement()
      && offset + length <= bufferLength / tLength())
    val tOffset = src.storageOffset() - 1 + srcOffset
    UnCompressedTensor.tensorToBytes(src, tOffset,
      buffer, bufferOffset + offset, length)
    this
  }

  override def compress(src: Tensor[T]): this.type = compress(0, src, 0, src.nElement())

  override def bytes(): ByteBuffer = bytes(0, bufferLength / tLength())

  override def bytes(offset: Int, length: Int): ByteBuffer = {
    require(offset >= 0 && length > 0 && (offset + length) * tLength() <= bufferLength,
      s"$offset $length $bufferLength")
    if (classTag[T] == classTag[Double]) {
      ByteBuffer.wrap(buffer, offset * tLength() + bufferOffset, length * tLength())
    } else if (classTag[T] == classTag[Float]) {
      ByteBuffer.wrap(buffer, offset * tLength() + bufferOffset, length * tLength())
    } else {
      throw new IllegalArgumentException
    }
  }

  override def deCompress(tensor: Tensor[T]): Unit = {
    deCompress(0, tensor, 0, bufferLength / tLength())
  }

  override def deCompress(srcOffset: Int, tensor: Tensor[T],
                          tgtOffset: Int, length: Int): Unit = {
    require(srcOffset >= 0 && length > 0 && (srcOffset + length) * tLength() <= bufferLength &&
      tgtOffset >= 0 && tgtOffset + length <= tensor.nElement())
    require(tensor.isContiguous())
    val toffset = tensor.storageOffset() - 1 + tgtOffset
    UnCompressedTensor.bytesToTensor[T](buffer, srcOffset * tLength() + bufferOffset,
          length * tLength(), tensor, toffset)
  }

  override def add(data: ByteBuffer): this.type = add(data, 0, bufferLength / tLength())

  override def add(data: ByteBuffer, offset: Int, length: Int): this.type = {
    require(offset >= 0 && length > 0 && (offset + length) * tLength() <= bufferLength)
    require(length * tLength() == data.remaining())
    UnCompressedTensor.add(buffer, offset * tLength() + bufferOffset,
      data.array(), data.position(), data.remaining())
    this
  }

  override def parAdd(data: ByteBuffer): this.type = add(data, 0, bufferLength / tLength())

  override def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type = {
    require(offset >= 0 && length > 0 && (offset + length) * tLength() <= bufferLength)
    require(length * tLength() == data.remaining())
    UnCompressedTensor.parAdd(buffer, offset * tLength() + bufferOffset, data.array(),
      data.position(), data.remaining())
    this
  }
}

object UnCompressedTensor {
  /**
   * The number of bytes of T
   * @tparam T Double or Float
   * @return
   */
  def tLength[T: ClassTag](): Int = {
    if (classTag[T] == classTag[Double]) {
      8
    } else {
      4
    }
  }

  private def parAdd[T: ClassTag](l: Array[Byte], lOffset: Int, r: Array[Byte],
                     rOffset: Int, length: Int): Array[Byte] = {
    require(length % tLength() == 0)
    require(lOffset + length <= l.length)
    require(rOffset + length <= r.length)

    val elementSize = length / tLength()
    val taskSize = elementSize / Engine.coreNumber()
    val extraSize = elementSize % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()

    if (classTag[T] == classTag[Double]) {
      val lBuffer = java.nio.ByteBuffer.wrap(l).asDoubleBuffer()
      val rBuffer = java.nio.ByteBuffer.wrap(r).asDoubleBuffer()
      Engine.default.invokeAndWait(
        (0 until availableTask).map(tid => () => {
          val start = tid * taskSize * tLength() + math.min(extraSize, tid) * tLength()
          val end = (tid + 1) * taskSize * tLength() + math.min(extraSize, tid + 1) * tLength()
          var i = start / tLength()
          while (i < end / tLength()) {
            lBuffer.put(i + lOffset / tLength(),
              lBuffer.get(i + lOffset / tLength()) + rBuffer.get(i + rOffset / tLength()))
            i += 1
          }

        })
      )
    } else if (classTag[T] == classTag[Float]) {
      val lBuffer = java.nio.ByteBuffer.wrap(l).asFloatBuffer()
      val rBuffer = java.nio.ByteBuffer.wrap(r).asFloatBuffer()
      Engine.default.invokeAndWait(
        (0 until availableTask).map(tid => () => {
          val start = tid * taskSize * tLength() + math.min(extraSize, tid) * tLength()
          val end = (tid + 1) * taskSize * tLength() + math.min(extraSize, tid + 1) * tLength()
          var i = start / tLength()
          while (i < end / tLength()) {
            lBuffer.put(i + lOffset / tLength(),
              lBuffer.get(i + lOffset / tLength()) + rBuffer.get(i + rOffset / tLength()))
            i += 1
          }

        })
      )
    } else {
      throw new IllegalArgumentException
    }

    l
  }

  private def add[T: ClassTag](l: Array[Byte], lOffset: Int, r: Array[Byte],
                  rOffset: Int, length: Int): Array[Byte] = {
    require(length % tLength() == 0)
    require(lOffset + length <= l.length)
    require(rOffset + length <= r.length)

    if (classTag[T] == classTag[Double]) {
      val lBuffer = java.nio.ByteBuffer.wrap(l).asDoubleBuffer()
      val rBuffer = java.nio.ByteBuffer.wrap(r).asDoubleBuffer()
      var i = 0
      while (i < length / tLength()) {
        lBuffer.put(i + lOffset/tLength(), lBuffer.get(i + lOffset/tLength()) +
          rBuffer.get(i + rOffset / tLength()))
        i += 1
      }
    } else if (classTag[T] == classTag[Float]) {
      val lBuffer = java.nio.ByteBuffer.wrap(l).asFloatBuffer()
      val rBuffer = java.nio.ByteBuffer.wrap(r).asFloatBuffer()
      var i = 0
      while (i < length / tLength()) {
        lBuffer.put(i + lOffset/tLength(), lBuffer.get(i + lOffset/tLength()) +
          rBuffer.get(i + rOffset / tLength()))
        i += 1
      }
    } else {
      throw new IllegalArgumentException
    }

    l
  }

  private[parameters] def tensorToBytes[T: ClassTag](
      src: Tensor[T],
      srcOffset: Int,
      tgt: Array[Byte],
      tgtOffset: Int,
      length: Int): Array[Byte] = {
    require(srcOffset + length <= src.nElement())
    require(tgtOffset + length * 2 <= tgt.length)

    val taskSize = length / Engine.coreNumber()
    val extraSize = length % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()

    if (classTag[T] == classTag[Double]) {
      val tgtBuffer = java.nio.ByteBuffer.wrap(tgt).asDoubleBuffer()
      val srcArray = src.storage().array().asInstanceOf[Array[Double]]
      Engine.default.invokeAndWait(
        (0 until availableTask).map(tid => () => {
          val start = tid * taskSize + math.min(extraSize, tid)
          val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
          var i = start
          while (i < end) {
            tgtBuffer.put(tgtOffset + i, srcArray(i + srcOffset))
            i += 1
          }
        })
      )
    } else if (classTag[T] == classTag[Float]) {
      val tgtBuffer = java.nio.ByteBuffer.wrap(tgt).asFloatBuffer()
      val srcArray = src.storage().array().asInstanceOf[Array[Float]]
      Engine.default.invokeAndWait(
        (0 until availableTask).map(tid => () => {
          val start = tid * taskSize + math.min(extraSize, tid)
          val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
          var i = start
          while (i < end) {
            tgtBuffer.put(tgtOffset + i, srcArray(i + srcOffset))
            i += 1
          }
        })
      )
    } else {
      throw new IllegalArgumentException
    }

    tgt
  }

  private[parameters] def bytesToTensor[T: ClassTag](
      src: Array[Byte],
      srcOffset: Int,
      srcLength: Int,
      target: Tensor[T],
      targetOffset: Int): Unit = {
    require(srcLength % tLength() == 0)
    require(srcLength + srcOffset <= src.length)
    require(srcLength / tLength() + targetOffset <= target.nElement())

    val targetLength = srcLength / tLength()
    val taskSize = targetLength / Engine.coreNumber()
    val extraSize = targetLength % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()

    if (classTag[T] == classTag[Double]) {
      val srcBuffer = java.nio.ByteBuffer.wrap(src).asDoubleBuffer()
      val targetArray = target.storage().array().asInstanceOf[Array[Double]]
      Engine.default.invokeAndWait(
        (0 until availableTask).map(tid => () => {
          val start = tid * taskSize * tLength() + math.min(extraSize, tid) * tLength()
          val end = (tid + 1) * taskSize * tLength() + math.min(extraSize, tid + 1) * tLength()
          var i = start
          while (i < end && i < srcLength + srcOffset) {
            targetArray(i / tLength() + targetOffset) =
              srcBuffer.get(i / tLength() + srcOffset / tLength())
            i += tLength()
          }
        })
      )
    } else if (classTag[T] == classTag[Float]) {
      val srcBuffer = java.nio.ByteBuffer.wrap(src).asFloatBuffer()
      val targetArray = target.storage().array().asInstanceOf[Array[Float]]
      Engine.default.invokeAndWait(
        (0 until availableTask).map(tid => () => {
          val start = tid * taskSize * tLength() + math.min(extraSize, tid) * tLength()
          val end = (tid + 1) * taskSize * tLength() + math.min(extraSize, tid + 1) * tLength()
          var i = start
          while (i < end && i < srcLength + srcOffset) {
            targetArray(i / tLength() + targetOffset) =
              srcBuffer.get(i / tLength() + srcOffset / tLength())
            i += tLength()
          }
        })
      )
    } else {
      throw new IllegalArgumentException
    }
  }

  private[parameters] def tensorToBytes[T: ClassTag](
      src: Tensor[T],
      tgt: Array[Byte],
      tgtOffset: Int): Array[Byte] = {
    require(src.isContiguous())
    if (src.getType() == DoubleType) {
      val arr = src.storage().array().asInstanceOf[Array[Double]]
      val arrOffset = src.storageOffset() - 1
      val dBuf = java.nio.ByteBuffer.wrap(tgt).asDoubleBuffer()
      var i = tgtOffset / tLength()
      while (i < src.nElement()) {
        dBuf.put(i, arr(i + arrOffset))
        i += 1
      }
    } else if (src.getType() == FloatType) {
      val arr = src.storage().array().asInstanceOf[Array[Float]]
      val arrOffset = src.storageOffset() - 1
      val fBuf = java.nio.ByteBuffer.wrap(tgt).asFloatBuffer()
      var i = tgtOffset / tLength()
      while (i < src.nElement()) {
        fBuf.put(i, arr(i + arrOffset))
        i += 1
      }
    } else {
      throw new IllegalArgumentException(s"Unknown tensor type ${src.getType()}")
    }
    tgt
  }

  def bytesToTensor[T: ClassTag](
      src: Array[Byte],
      srcOffset: Int,
      target: Tensor[T]): Tensor[T] = {
    require(target.isContiguous())
    if (target.getType() == DoubleType) {
      val arr = target.storage().array().asInstanceOf[Array[Double]]
      val arrOffset = target.storageOffset() - 1
      val dBuf = java.nio.ByteBuffer.wrap(src).asDoubleBuffer()
      var i = srcOffset / tLength()
      while (i < target.nElement()) {
        arr(i + arrOffset) = dBuf.get(i)
        i += 1
      }
    } else if (target.getType() == FloatType) {
      val arr = target.storage().array().asInstanceOf[Array[Float]]
      val arrOffset = target.storageOffset() - 1
      val dBuf = java.nio.ByteBuffer.wrap(src).asFloatBuffer()
      var i = srcOffset / tLength()
      while (i < target.nElement()) {
        arr(i + arrOffset) = dBuf.get(i)
        i += 1
      }
    } else {
      throw new IllegalArgumentException(s"Unknown tensor type ${target.getType()}")
    }
    target
  }
}

