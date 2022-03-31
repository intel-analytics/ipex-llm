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

package com.intel.analytics.bigdl.dllib.optim.parameters

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{Engine, Log4Error}

import scala.reflect._

private[bigdl] class UncompressedTensor[T: ClassTag](
                                                        buffer: Array[Byte],
                                                        bufferOffset: Int,
                                                        bufferLength: Int)
  extends CompressedTensor[T] {

  def this(tensor: Tensor[T]) {
    this(new Array[Byte](tensor.nElement() * 4), 0, tensor.nElement() * 4)
    compress(tensor)
  }

  def this(length: Int) = this(new Array[Byte](length * 4), 0, length * 4)

  def this(bytes: ByteBuffer) = this(bytes.array(), bytes.position(), bytes.remaining())

  Log4Error.unKnowExceptionError(bufferOffset + bufferLength <= buffer.length,
    s"bufferOffset $bufferOffset + bufferLength $bufferLength should not greater" +
      s" than buffer.length ${buffer.length}")
  Log4Error.unKnowExceptionError(bufferLength % 4 == 0,
    s"bufferLength $bufferLength should not be able to divide by 4")

  override def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int)
  : this.type = {
    Log4Error.unKnowExceptionError(src.isContiguous(), s"src needs to be contiguous")
    Log4Error.unKnowExceptionError(offset >= 0 && srcOffset >= 0,
      s"offset cannot be negative, but got $offset. srcOffset cannot be negative," +
        s"but got $srcOffset.")
    Log4Error.unKnowExceptionError(srcOffset + length <= src.nElement(),
      s"srcOffset $srcOffset + length $length should not be greater than" +
        s"src.nElement() ${src.nElement()}")
    Log4Error.unKnowExceptionError(offset + length <= bufferLength / 4,
      s"offset $offset + length $length should not be greater than" +
        s" bufferLength ${bufferLength} / 4")

    val tOffset = src.storageOffset() - 1 + srcOffset
    if (classTag[T] == classTag[Double]) {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float")
    } else if (classTag[T] == classTag[Float]) {
      UncompressedTensor.toBytes(src.storage().array().asInstanceOf[Array[Float]], tOffset,
        buffer, bufferOffset + offset, length)
    } else {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float")
    }

    this
  }

  override def compress(src: Tensor[T]): this.type = compress(0, src, 0, src.nElement())

  override def bytes(): ByteBuffer = bytes(0, bufferLength / 4)

  override def bytes(offset: Int, length: Int): ByteBuffer = {
    Log4Error.unKnowExceptionError(offset >= 0 && length > 0
      && (offset + length) * 4 <= bufferLength,
      s"offset $offset cannot be negative and length $length cannot be negative," +
        s"(offset $offset + length $length) * 4 should not be greater than " +
        s"bufferLength $bufferLength")

    if (classTag[T] == classTag[Double]) {
      ByteBuffer.wrap(buffer, offset * 4 + bufferOffset, length * 4)
    } else if (classTag[T] == classTag[Float]) {
      ByteBuffer.wrap(buffer, offset * 4 + bufferOffset, length * 4)
    } else {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float and Double")
      null
    }
  }

  override def deCompress(tensor: Tensor[T]): Unit = deCompress(0, tensor, 0, bufferLength / 4)

  override def deCompress(srcOffset: Int, tensor: Tensor[T],
                          tgtOffset: Int, length: Int): Unit = {
    Log4Error.unKnowExceptionError(srcOffset >= 0 && length > 0 && tgtOffset >= 0,
      s"srcOffset $srcOffset, tgtOffset $tgtOffset, length $length cannot be negative")
    Log4Error.unKnowExceptionError((srcOffset + length) * 4 <= bufferLength,
      s"(srcOffset $srcOffset + length $length) * 4 should not be greater" +
        s" bufferLength $bufferLength")
    Log4Error.unKnowExceptionError(tgtOffset + length <= tensor.nElement(),
      s"tgtOffset $tgtOffset + length $length should not be greater than " +
        s"tensor.nElement() ${tensor.nElement()}")
    Log4Error.unKnowExceptionError(tensor.isContiguous(),
      "tensor is expected to be contiguous")

    if (classTag[T] == classTag[Double]) {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float")
    } else if (classTag[T] == classTag[Float]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Float]]
      val toffset = tensor.storageOffset() - 1 + tgtOffset
      UncompressedTensor.fromBytes(buffer, srcOffset * 4 + bufferOffset,
        length * 4, tdata, toffset)
    } else {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float")
    }
  }

  override def add(data: ByteBuffer): this.type = add(data, 0, bufferLength / 4)

  override def add(data: ByteBuffer, offset: Int, length: Int): this.type = {
    Log4Error.unKnowExceptionError(offset >= 0 && length > 0
      && (offset + length) * 4 <= bufferLength,
      s"offset $offset cannot be negative, length $length should be positive," +
        s" (offset $offset + length $length) * 4 should not be greater" +
        s" than bufferLength $bufferLength")
    Log4Error.unKnowExceptionError(length * 4 == data.remaining(),
      s"length $length * 4 doesn't match data.remaining() ${data.remaining()}")
    UncompressedTensor.add(buffer, offset * 4 + bufferOffset,
      data.array(), data.position(), data.remaining())
    this
  }

  override def parAdd(data: ByteBuffer): this.type = add(data, 0, bufferLength / 4)

  override def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type = {
    Log4Error.unKnowExceptionError(offset >= 0 && length > 0
      && (offset + length) * 4 <= bufferLength,
      s"offset $offset cannot be negative, length $length should be positive," +
        s" (offset $offset + length $length) * 4 should not be greater" +
        s" than bufferLength $bufferLength")
    Log4Error.unKnowExceptionError(length * 4 == data.remaining(),
      s"length $length * 4 should match data.remaining() ${data.remaining()}")

    UncompressedTensor.parAdd(buffer, offset * 4 + bufferOffset, data.array(),
      data.position(), data.remaining())
    this
  }
}

object UncompressedTensor {
  private def parAdd(l: Array[Byte], lOffset: Int, r: Array[Byte],
                     rOffset: Int, length: Int): Array[Byte] = {
    val start = System.nanoTime()
    Log4Error.unKnowExceptionError(length % 4 == 0,
      s"length $length should be able to divide by 4")
    Log4Error.unKnowExceptionError(lOffset + length <= l.length,
      s"lOffset $lOffset + length $length should not be greater than l.length ${l.length}")
    Log4Error.unKnowExceptionError(rOffset + length <= r.length,
      s"rOffset $rOffset + length $length should not be greater than r.length ${r.length}")

    val elementSize = length / 4
    val taskSize = elementSize / Engine.coreNumber()
    val extraSize = elementSize % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
        val start = tid * taskSize * 4 + math.min(extraSize, tid) * 4
        val end = (tid + 1) * taskSize * 4 + math.min(extraSize, tid + 1) * 4
        var i = start
        while (i < end) {
          val sum = toFloat(
            l(i + lOffset),
            l(i + lOffset + 1),
            l(i + lOffset + 2),
            l(i + lOffset + 3)) + toFloat(r(i + rOffset),
            r(i + rOffset + 1),
            r(i + rOffset + 2),
            r(i + rOffset + 3))
          val bytes = float2bytes(sum)
          l(i + lOffset) = (bytes >>> 24 & 0xff).toByte
          l(i + lOffset + 1) = (bytes >>> 16 & 0xff).toByte
          l(i + lOffset + 2) = (bytes >>> 8 & 0xff).toByte
          l(i + lOffset + 3) = (bytes >>> 0 & 0xff).toByte
          i += 4
        }
      })
    )

    l
  }

  private def add(l: Array[Byte], lOffset: Int, r: Array[Byte],
                  rOffset: Int, length: Int): Array[Byte] = {
    val start = System.nanoTime()
    Log4Error.unKnowExceptionError(length % 4 == 0,
      s"length $length should be able to divide by 4")
    Log4Error.unKnowExceptionError(lOffset + length <= l.length,
      s"lOffset $lOffset + length $length should not be greater than l.length ${l.length}")
    Log4Error.unKnowExceptionError(rOffset + length <= r.length,
      s"rOffset $rOffset + length $length should not be greater than r.length ${r.length}")

    var i = 0
    while (i < length) {
      val sum = toFloat(
        l(i + lOffset),
        l(i + lOffset + 1),
        l(i + lOffset + 2),
        l(i + lOffset + 3)) + toFloat(r(i + rOffset),
        r(i + rOffset + 1),
        r(i + rOffset + 2),
        r(i + rOffset + 3))
      val bytes = float2bytes(sum)
      l(i + lOffset) = (bytes >>> 24 & 0xff).toByte
      l(i + lOffset + 1) = (bytes >>> 16 & 0xff).toByte
      l(i + lOffset + 2) = (bytes >>> 8 & 0xff).toByte
      l(i + lOffset + 3) = (bytes >>> 0 & 0xff).toByte

      i += 4
    }

    l
  }

  private[parameters] def toBytes(src: Array[Float], srcOffset: Int, tgt: Array[Byte],
                                 tgtOffset: Int, length: Int): Array[Byte] = {
    Log4Error.unKnowExceptionError(srcOffset + length <= src.length,
      s"srcOffset $srcOffset + length $length should not greater than src.length ${src.length}")
    Log4Error.unKnowExceptionError(tgtOffset + length * 4 <= tgt.length,
      s"tgtOffset $tgtOffset + 4 * length $length should not greater than" +
        s"tgt.length ${tgt.length}")

    val taskSize = length / Engine.coreNumber()
    val extraSize = length % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
        val start = tid * taskSize + math.min(extraSize, tid)
        val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
        var i = start
        while (i < end) {
          val bytes = float2bytes(src(i + srcOffset))
          tgt(tgtOffset + i * 4) = (bytes >>> 24 & 0xff).toByte
          tgt(tgtOffset + i * 4 + 1) = (bytes >>> 16 & 0xff).toByte
          tgt(tgtOffset + i * 4 + 2) = (bytes >>> 8 & 0xff).toByte
          tgt(tgtOffset + i * 4 + 3) = (bytes & 0xff).toByte
          i += 1
        }
      })
    )

    tgt
  }


  private[parameters] def toBytes(src: Array[Double], srcOffset: Int, tgt: Array[Byte],
                                 tgtOffset: Int, length: Int): Array[Byte] = {
    Log4Error.unKnowExceptionError(srcOffset + length <= src.length,
      s"srcOffset $srcOffset + length $length should not greater than src.length ${src.length}")
    Log4Error.unKnowExceptionError(tgtOffset + length * 4 <= tgt.length,
      s"tgtOffset $tgtOffset + 4 * length $length should not greater than" +
        s"tgt.length ${tgt.length}")

    val taskSize = length / Engine.coreNumber()
    val extraSize = length % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
        val start = tid * taskSize + math.min(extraSize, tid)
        val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
        var i = start
        while (i < end) {
          val bytes = float2bytes(src(i + srcOffset).toFloat)
          tgt(tgtOffset + i * 4) = (bytes >>> 24 & 0xff).toByte
          tgt(tgtOffset + i * 4 + 1) = (bytes >>> 16 & 0xff).toByte
          tgt(tgtOffset + i * 4 + 2) = (bytes >>> 8 & 0xff).toByte
          tgt(tgtOffset + i * 4 + 3) = (bytes & 0xff).toByte
          i += 1
        }
      })
    )

    tgt
  }

  private[parameters] def fromBytes(bytes: Array[Byte], bytesOffset: Int, bytesLength: Int,
                                    target: Array[Float], targetOffset: Int): Unit = {
    Log4Error.unKnowExceptionError(bytesLength % 4 == 0,
      s"bytesLength should be able to divide by 4, but got $bytesLength")
    Log4Error.unKnowExceptionError(bytesLength + bytesOffset <= bytes.length,
      s"bytesLength $bytesLength + bytesOffset $bytesOffset should not greater" +
        s" than bytes.length ${bytes.length}")
    Log4Error.unKnowExceptionError(bytesLength / 4 + targetOffset <= target.length,
      s"bytesLength $bytesLength / 4 + targetOffset $targetOffset should not greater" +
        s" than target.length ${target.length}")

    val targetLength = bytesLength / 4
    val taskSize = targetLength / Engine.coreNumber()
    val extraSize = targetLength % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
        val start = tid * taskSize * 4 + math.min(extraSize, tid) * 4
        val end = (tid + 1) * taskSize * 4 + math.min(extraSize, tid + 1) * 4
        var i = start
        while (i < end && i < bytesLength + bytesOffset) {
          target(i / 4 + targetOffset) = toFloat(
            bytes(i + bytesOffset),
            bytes(i + 1 + bytesOffset),
            bytes(i + 2 + bytesOffset),
            bytes(i + 3 + bytesOffset))
          i += 4
        }
      })
    )
  }

  @inline
  private def float2bytes(value: Float): Int = {
    java.lang.Float.floatToRawIntBits(value)
  }

  @inline
  private def toFloat(byte1: Byte, byte2: Byte, byte3: Byte, byte4: Byte): Float = {
    java.lang.Float.intBitsToFloat(byte1 << 24 |
      (byte2 << 16 & 0x00ff0000) |
      (byte3 << 8 & 0x0000ff00) |
      (byte4 & 0x000000ff))
  }
}
