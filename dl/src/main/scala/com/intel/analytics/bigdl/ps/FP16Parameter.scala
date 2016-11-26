/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.ps

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect._

class FP16Parameter[T: ClassTag](buffer: Array[Byte], bufferOffset: Int, bufferLength: Int)
  extends Parameter[T] {

  def this(tensor: Tensor[T]) {
    this(new Array[Byte](tensor.nElement() * 2), 0, tensor.nElement() * 2)
    copyFrom(tensor)
  }

  def this(length: Int) = this(new Array[Byte](length * 2), 0, length * 2)

  def this(bytes: ByteBuffer) = this(bytes.array(), bytes.position(), bytes.remaining())

  require(bufferLength % 2 == 0 && bufferOffset + bufferLength <= buffer.length)

  override def copyFrom(offset: Int, src: Tensor[T], srcOffset: Int, length: Int)
  : this.type = {
    require(src.isContiguous() && offset >= 0 && srcOffset >= 0 &&
      srcOffset + length <= src.nElement()
      && offset + length <= bufferLength / 2)
    val tOffset = src.storageOffset() - 1 + srcOffset
    if (classTag[T] == classTag[Double]) {
      FP16Parameter.toFP16(src.storage().array().asInstanceOf[Array[Double]], tOffset,
        buffer, bufferOffset + offset, length)
    } else if (classTag[T] == classTag[Float]) {
      FP16Parameter.toFP16(src.storage().array().asInstanceOf[Array[Float]], tOffset,
        buffer, bufferOffset + offset, length)
    } else {
      throw new IllegalArgumentException
    }

    this
  }

  override def copyFrom(src: Tensor[T]): this.type = copyFrom(0, src, 0, src.nElement())

  override def bytes(): ByteBuffer = bytes(0, bufferLength / 2)

  override def bytes(offset: Int, length: Int): ByteBuffer = {
    require(offset >= 0 && length > 0 && (offset + length) * 2 <= bufferLength,
      s"$offset $length $bufferLength")
    if (classTag[T] == classTag[Double]) {
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    } else if (classTag[T] == classTag[Float]) {
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    } else {
      throw new IllegalArgumentException
    }
  }

  override def copyTo(tensor: Tensor[T]): Unit = copyTo(0, tensor, 0, bufferLength / 2)

  override def copyTo(srcOffset: Int, tensor: Tensor[T],
    tgtOffset: Int, length: Int): Unit = {
    require(srcOffset >= 0 && length > 0 && (srcOffset + length) * 2 <= bufferLength &&
      tgtOffset >= 0 && tgtOffset + length <= tensor.nElement())
    require(tensor.isContiguous())
    if (classTag[T] == classTag[Double]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Double]]
      val toffset = tensor.storageOffset() - 1 + tgtOffset
      FP16Parameter.fromFP16(buffer, srcOffset * 2 + bufferOffset, length * 2, tdata, toffset)
    } else if (classTag[T] == classTag[Float]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Float]]
      val toffset = tensor.storageOffset() - 1 + tgtOffset
      FP16Parameter.fromFP16(buffer, srcOffset * 2 + bufferOffset, length * 2, tdata, toffset)
    } else {
      throw new IllegalArgumentException
    }
  }

  override def add(data: ByteBuffer): this.type = add(data, 0, bufferLength / 2)

  override def add(data: ByteBuffer, offset: Int, length: Int): this.type = {
    require(offset >= 0 && length > 0 && (offset + length) * 2 <= bufferLength)
    require(length * 2 == data.remaining())
    FP16Parameter.add(buffer, offset * 2 + bufferOffset,
      data.array(), data.position(), data.remaining())
    this
  }

  override def parAdd(data: ByteBuffer): this.type = add(data, 0, bufferLength / 2)

  override def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type = {
    require(offset >= 0 && length > 0 && (offset + length) * 2 <= bufferLength)
    require(length * 2 == data.remaining())
    FP16Parameter.parAdd(buffer, offset * 2 + bufferOffset, data.array(),
      data.position(), data.remaining())
    this
  }
}

object FP16Parameter {
  private def parAdd(l: Array[Byte], lOffset: Int, r: Array[Byte],
    rOffset: Int, length: Int): Array[Byte] = {
    val start = System.nanoTime()
    require(length % 2 == 0)
    require(lOffset + length <= l.length)
    require(rOffset + length <= r.length)

    val elementSize = length / 2
    val taskSize = elementSize / Engine.coresNum
    val extraSize = elementSize % Engine.coresNum
    val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
    (0 until availableTask).map(tid => Future {
      val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
      val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
      var i = start
      while (i < end) {
        val sum = toFloat(l(i + lOffset), l(i + lOffset + 1)) +
          toFloat(r(i + rOffset), r(i + rOffset + 1))
        val bytes = truncate(sum)
        l(i + lOffset) = (bytes >>> 24 & 0xff).toByte
        l(i + lOffset + 1) = (bytes >>> 16 & 0xff).toByte
        i += 2
      }
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))

    l
  }

  private def add(l: Array[Byte], lOffset: Int, r: Array[Byte],
    rOffset: Int, length: Int): Array[Byte] = {
    val start = System.nanoTime()
    require(length % 2 == 0)
    require(lOffset + length <= l.length)
    require(rOffset + length <= r.length)

    var i = 0
    while (i < length) {
      val sum = toFloat(l(i + lOffset), l(i + lOffset + 1)) +
        toFloat(r(i + rOffset), r(i + rOffset + 1))
      val bytes = truncate(sum)
      l(i + lOffset) = (bytes >>> 24 & 0xff).toByte
      l(i + lOffset + 1) = (bytes >>> 16 & 0xff).toByte

      i += 2
    }

    l
  }

  private[ps] def toFP16(src: Array[Float], srcOffset: Int, tgt: Array[Byte],
    tgtOffset: Int, length: Int): Array[Byte] = {
    require(srcOffset + length <= src.length)
    require(tgtOffset + length * 2 <= tgt.length)

    val taskSize = length / Engine.coresNum
    val extraSize = length % Engine.coresNum
    val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
    (0 until availableTask).map(tid => Future {
      val start = tid * taskSize + math.min(extraSize, tid)
      val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
      var i = start
      while (i < end) {
        val bytes = truncate(src(i + srcOffset))
        tgt(tgtOffset + i * 2) = (bytes >>> 24 & 0xff).toByte
        tgt(tgtOffset + i * 2 + 1) = (bytes >>> 16 & 0xff).toByte
        i += 1
      }
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))

    tgt
  }


  private[ps] def toFP16(src: Array[Double], srcOffset: Int, tgt: Array[Byte],
    tgtOffset: Int, length: Int): Array[Byte] = {
    require(srcOffset + length <= src.length)
    require(tgtOffset + length * 2 <= tgt.length)

    val taskSize = length / Engine.coresNum
    val extraSize = length % Engine.coresNum
    val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
    (0 until availableTask).map(tid => Future {
      val start = tid * taskSize + math.min(extraSize, tid)
      val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
      var i = start
      while (i < end) {
        val bytes = truncate(src(i + srcOffset).toFloat)
        tgt(tgtOffset + i * 2) = (bytes >>> 24 & 0xff).toByte
        tgt(tgtOffset + i * 2 + 1) = (bytes >>> 16 & 0xff).toByte
        i += 1
      }
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))

    tgt
  }

  private[ps] def fromFP16(fp16: Array[Byte], fp16Offset: Int, fp16Length: Int,
    target: Array[Float], targetOffset: Int): Unit = {
    require(fp16Length % 2 == 0)
    require(fp16Length + fp16Offset <= fp16.length)
    require(fp16Length / 2 + targetOffset <= target.length)

    val targetLength = fp16Length / 2
    val taskSize = targetLength / Engine.coresNum
    val extraSize = targetLength % Engine.coresNum
    val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
    (0 until availableTask).map(tid => Future {
      val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
      val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
      var i = start
      while (i < end && i < fp16Length + fp16Offset) {
        target(i / 2 + targetOffset) = toFloat(fp16(i + fp16Offset), fp16(i + 1 + fp16Offset))
        i += 2
      }
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))
  }

  private[ps] def fromFP16(fp16: Array[Byte], fp16Offset: Int, fp16Length: Int,
    target: Array[Double], targetOffset: Int): Unit = {
    require(fp16Length % 2 == 0)
    require(fp16Length + fp16Offset <= fp16.length)
    require(fp16Length / 2 + targetOffset <= target.length)

    val targetLength = fp16Length / 2
    val taskSize = targetLength / Engine.coresNum
    val extraSize = targetLength % Engine.coresNum
    val availableTask = if (taskSize == 0) extraSize else Engine.coresNum
    (0 until availableTask).map(tid => Future {
      val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
      val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
      var i = start
      while (i < end && i < fp16Length + fp16Offset) {
        target(i / 2 + targetOffset) = toFloat(fp16(i + fp16Offset), fp16(i + 1 + fp16Offset))
        i += 2
      }
    }(Engine.getInstance())).map(Await.result(_, Duration.Inf))
  }

  @inline
  private def truncate(value: Float): Int = {
    java.lang.Float.floatToRawIntBits(value) & 0xffff0000
  }

  @inline
  private def toFloat(byte1: Byte, byte2: Byte): Float = {
    java.lang.Float.intBitsToFloat(byte1 << 24 | byte2 << 16 & 0x00ff0000)
  }
}
