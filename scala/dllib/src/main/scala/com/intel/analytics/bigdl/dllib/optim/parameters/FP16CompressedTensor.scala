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

private[bigdl] class FP16CompressedTensor[T: ClassTag](
      buffer: Array[Byte],
      bufferOffset: Int,
      bufferLength: Int)
  extends CompressedTensor[T] {

  def this(tensor: Tensor[T]) {
    this(new Array[Byte](tensor.nElement() * 2), 0, tensor.nElement() * 2)
    compress(tensor)
  }

  def this(length: Int) = this(new Array[Byte](length * 2), 0, length * 2)

  def this(bytes: ByteBuffer) = this(bytes.array(), bytes.position(), bytes.remaining())

  Log4Error.unKnowExceptionError(bufferLength % 2 == 0 &&
    bufferOffset + bufferLength <= buffer.length,
  s"bufferLength needs to be even, but is $bufferLength, bufferOffset $bufferOffset" +
    s" + bufferLength $bufferLength should not be greater than buffer.length ${buffer.length}")

  override def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int)
  : this.type = {
    Log4Error.unKnowExceptionError(src.isContiguous(), s"src needs to be contiguous")
    Log4Error.unKnowExceptionError(offset >= 0 && srcOffset >= 0,
      s"offset cannot be negative, but got $offset. srcOffset cannot be negative," +
        s"but got $srcOffset.")
    Log4Error.unKnowExceptionError(srcOffset + length <= src.nElement(),
      s"srcOffset $srcOffset + length $length should not be greater than" +
        s"src.nElement() ${src.nElement()}")
    Log4Error.unKnowExceptionError(offset + length <= bufferLength / 2,
      s"offset $offset + length $length should not be greater than" +
        s" bufferLength / 2 ${bufferLength}")
    val tOffset = src.storageOffset() - 1 + srcOffset
    if (classTag[T] == classTag[Double]) {
      FP16CompressedTensor.toFP16(src.storage().array().asInstanceOf[Array[Double]], tOffset,
        buffer, bufferOffset + offset, length)
    } else if (classTag[T] == classTag[Float]) {
      FP16CompressedTensor.toFP16(src.storage().array().asInstanceOf[Array[Float]], tOffset,
        buffer, bufferOffset + offset, length)
    } else {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float and Double")
    }

    this
  }

  override def compress(src: Tensor[T]): this.type = compress(0, src, 0, src.nElement())

  override def bytes(): ByteBuffer = bytes(0, bufferLength / 2)

  override def bytes(offset: Int, length: Int): ByteBuffer = {
    Log4Error.unKnowExceptionError(offset >= 0 && length > 0
      && (offset + length) * 2 <= bufferLength,
      s"offset $offset cannot be negative and length $length cannot be negative," +
        s"(offset $offset + length $length) * 2 should not be greater than " +
        s"bufferLength $bufferLength")
    if (classTag[T] == classTag[Double]) {
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    } else if (classTag[T] == classTag[Float]) {
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    } else {
      Log4Error.unKnowExceptionError(false, s"${classTag[T]} is not supported",
        "only support Float and Double")
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    }
  }

  override def deCompress(tensor: Tensor[T]): Unit = deCompress(0, tensor, 0, bufferLength / 2)

  override def deCompress(srcOffset: Int, tensor: Tensor[T],
                          tgtOffset: Int, length: Int): Unit = {
    Log4Error.unKnowExceptionError(srcOffset >= 0 && length > 0 && tgtOffset >= 0,
      s"srcOffset $srcOffset, tgtOffset $tgtOffset, length $length cannot be negative")
    Log4Error.unKnowExceptionError((srcOffset + length) * 2 <= bufferLength,
      s"(srcOffset $srcOffset + length $length) * 2 should not be greater" +
        s" bufferLength $bufferLength")
    Log4Error.unKnowExceptionError(tgtOffset + length <= tensor.nElement(),
      s"tgtOffset $tgtOffset + length $length should not be greater than " +
        s"tensor.nElement() ${tensor.nElement()}")
    Log4Error.unKnowExceptionError(tensor.isContiguous(),
      "tensor is expected to be contiguous")
    if (classTag[T] == classTag[Double]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Double]]
      val toffset = tensor.storageOffset() - 1 + tgtOffset
      FP16CompressedTensor.fromFP16(buffer, srcOffset * 2 + bufferOffset,
        length * 2, tdata, toffset)
    } else if (classTag[T] == classTag[Float]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Float]]
      val toffset = tensor.storageOffset() - 1 + tgtOffset
      FP16CompressedTensor.fromFP16(buffer, srcOffset * 2 + bufferOffset,
        length * 2, tdata, toffset)
    } else {
      Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
        "only support Float and Double")
    }
  }

  override def add(data: ByteBuffer): this.type = add(data, 0, bufferLength / 2)

  override def add(data: ByteBuffer, offset: Int, length: Int): this.type = {
    Log4Error.unKnowExceptionError(offset >= 0 && length > 0
      && (offset + length) * 2 <= bufferLength,
      s"offset $offset cannot be negative, length $length should be positive," +
        s" (offset $offset + length $length) * 2 should not be greater" +
        s" than bufferLength $bufferLength")
    Log4Error.unKnowExceptionError(length * 2 == data.remaining(),
      s"length $length * 2 doesn't match data.remaining() ${data.remaining()}")
    FP16CompressedTensor.add(buffer, offset * 2 + bufferOffset,
      data.array(), data.position(), data.remaining())
    this
  }

  override def parAdd(data: ByteBuffer): this.type = add(data, 0, bufferLength / 2)

  override def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type = {
    Log4Error.unKnowExceptionError(offset >= 0 && length > 0
      && (offset + length) * 2 <= bufferLength,
      s"offset $offset cannot be negative, length $length should be positive," +
        s" (offset $offset + length $length) * 2 should not be greater" +
        s" than bufferLength $bufferLength")
    Log4Error.unKnowExceptionError(length * 2 == data.remaining(),
      s"length $length * 2 should match data.remaining() ${data.remaining()}")
    FP16CompressedTensor.parAdd(buffer, offset * 2 + bufferOffset, data.array(),
      data.position(), data.remaining())
    this
  }
}

object FP16CompressedTensor {
  private def parAdd(l: Array[Byte], lOffset: Int, r: Array[Byte],
    rOffset: Int, length: Int): Array[Byte] = {
    val start = System.nanoTime()
    Log4Error.unKnowExceptionError(length % 2 == 0, s"length $length should be even")
    Log4Error.unKnowExceptionError(lOffset + length <= l.length,
      s"lOffset $lOffset + length $length should not be greater than l.length ${l.length}")
    Log4Error.unKnowExceptionError(rOffset + length <= r.length,
      s"rOffset $rOffset + length $length should not be greater than r.length ${r.length}")

    val elementSize = length / 2
    val taskSize = elementSize / Engine.coreNumber()
    val extraSize = elementSize % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
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
      })
    )

    l
  }

  private def add(l: Array[Byte], lOffset: Int, r: Array[Byte],
    rOffset: Int, length: Int): Array[Byte] = {
    val start = System.nanoTime()
    Log4Error.unKnowExceptionError(length % 2 == 0, s"length $length should be even")
    Log4Error.unKnowExceptionError(lOffset + length <= l.length,
      s"lOffset $lOffset + length $length should not be greater than l.length ${l.length}")
    Log4Error.unKnowExceptionError(rOffset + length <= r.length,
      s"rOffset $rOffset + length $length should not be greater than r.length ${r.length}")

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

  private[parameters] def toFP16(src: Array[Float], srcOffset: Int, tgt: Array[Byte],
                                 tgtOffset: Int, length: Int): Array[Byte] = {
    Log4Error.unKnowExceptionError(srcOffset + length <= src.length,
      s"srcOffset $srcOffset + length $length should not greater than src.length ${src.length}")
    Log4Error.unKnowExceptionError(tgtOffset + length * 2 <= tgt.length,
      s"tgtOffset $tgtOffset + 2 * length $length should not greater than" +
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
          val bytes = truncate(src(i + srcOffset))
          tgt(tgtOffset + i * 2) = (bytes >>> 24 & 0xff).toByte
          tgt(tgtOffset + i * 2 + 1) = (bytes >>> 16 & 0xff).toByte
          i += 1
        }
      })
    )

    tgt
  }


  private[parameters] def toFP16(src: Array[Double], srcOffset: Int, tgt: Array[Byte],
                                 tgtOffset: Int, length: Int): Array[Byte] = {
    Log4Error.unKnowExceptionError(srcOffset + length <= src.length,
      s"srcOffset $srcOffset + length $length should not greater than src.length ${src.length}")
    Log4Error.unKnowExceptionError(tgtOffset + length * 2 <= tgt.length,
      s"tgtOffset $tgtOffset + 2 * length $length should not greater than" +
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
          val bytes = truncate(src(i + srcOffset).toFloat)
          tgt(tgtOffset + i * 2) = (bytes >>> 24 & 0xff).toByte
          tgt(tgtOffset + i * 2 + 1) = (bytes >>> 16 & 0xff).toByte
          i += 1
        }
      })
    )

    tgt
  }

  private[parameters] def fromFP16(fp16: Array[Byte], fp16Offset: Int, fp16Length: Int,
                                   target: Array[Float], targetOffset: Int): Unit = {
    Log4Error.unKnowExceptionError(fp16Length % 2 == 0,
      s"fp16Length should be even, but got $fp16Length")
    Log4Error.unKnowExceptionError(fp16Length + fp16Offset <= fp16.length,
      s"fp16Length $fp16Length + fp16Offset $fp16Offset should not greater" +
        s" than fp16.length ${fp16.length}")
    Log4Error.unKnowExceptionError(fp16Length / 2 + targetOffset <= target.length,
      s"fp16Length $fp16Length / 2 + targetOffset $targetOffset should not greater" +
        s" than target.length ${target.length}")

    val targetLength = fp16Length / 2
    val taskSize = targetLength / Engine.coreNumber()
    val extraSize = targetLength % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
        val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
        val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
        var i = start
        while (i < end && i < fp16Length + fp16Offset) {
          target(i / 2 + targetOffset) = toFloat(fp16(i + fp16Offset), fp16(i + 1 + fp16Offset))
          i += 2
        }
      })
    )
  }

  private[parameters] def fromFP16(fp16: Array[Byte], fp16Offset: Int, fp16Length: Int,
                                   target: Array[Double], targetOffset: Int): Unit = {
    Log4Error.unKnowExceptionError(fp16Length % 2 == 0,
      s"fp16Length should be even, but got $fp16Length")
    Log4Error.unKnowExceptionError(fp16Length + fp16Offset <= fp16.length,
      s"fp16Length $fp16Length + fp16Offset $fp16Offset should not greater" +
        s" than fp16.length ${fp16.length}")
    Log4Error.unKnowExceptionError(fp16Length / 2 + targetOffset <= target.length,
      s"fp16Length $fp16Length / 2 + targetOffset $targetOffset should not greater" +
        s" than target.length ${target.length}")

    val targetLength = fp16Length / 2
    val taskSize = targetLength / Engine.coreNumber()
    val extraSize = targetLength % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid => () => {
        val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
        val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
        var i = start
        while (i < end && i < fp16Length + fp16Offset) {
          target(i / 2 + targetOffset) = toFloat(fp16(i + fp16Offset), fp16(i + 1 + fp16Offset))
          i += 2
        }
      })
    )
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
