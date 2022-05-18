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
import com.intel.analytics.bigdl.dllib.utils.Log4Error

import scala.reflect._

class FP16SplitsCompressedTensor[T: ClassTag](buffers: Array[Array[Byte]], size: Int)
  extends CompressedTensor[T] {

  def this(tensor: Tensor[T], splitsNum: Int) {
    this(new Array[Array[Byte]](splitsNum), tensor.nElement())
    compress(tensor)
  }

  def this(length: Int, splitsNum: Int) {
    this(new Array[Array[Byte]](splitsNum), length)
  }

  @inline
  private def overlap(splitOffset: Int, splitLength: Int, offset: Int,
    length: Int): Option[(Int, Int)] = {
    if ((splitOffset > offset + length || splitOffset + splitLength < offset)) {
      None
    } else {
      Some(math.max(offset - splitOffset, 0),
        math.min(splitOffset + splitLength, offset + length) - math.max(splitOffset, offset))
    }
  }

  override def compress(offset: Int, src: Tensor[T], srcOffset: Int,
                        length: Int): FP16SplitsCompressedTensor.this.type = {
    Log4Error.unKnowExceptionError(src.isContiguous(),
      "src is expected to be contiguous")
    Log4Error.unKnowExceptionError(offset >= 0 && srcOffset >= 0 &&
      srcOffset + length <= src.nElement(),
      s"offset $offset cannot be negative, srcOffset $srcOffset cannot be negative," +
        s" srcOffset $srcOffset + length $length <= src.nElement() ${src.nElement()}")
    Log4Error.unKnowExceptionError(offset + length <= size,
      s"offset $offset + length $length should not" +
      s" greater than size $size")
    val tOffset = src.storageOffset() - 1 + srcOffset

    val splitSize = size / buffers.length
    val extraSize = size % buffers.length
    var i = 0
    while (i < buffers.length) {
      val start = splitSize * i + math.min(extraSize, i)
      val curLength = splitSize + (if (i < extraSize) 1 else 0)
      overlap(start, curLength, offset, length) match {
        case Some((splitOffset, overlapLength)) =>
          if (buffers(i) == null) {
            buffers(i) = new Array[Byte](curLength * 2)
          }
          if (classTag[T] == classTag[Double]) {
            FP16CompressedTensor.toFP16(src.storage().array().asInstanceOf[Array[Double]],
              tOffset + start, buffers(i), splitOffset, overlapLength)
          } else if (classTag[T] == classTag[Float]) {
            FP16CompressedTensor.toFP16(src.storage().array().asInstanceOf[Array[Float]],
              tOffset + start, buffers(i), splitOffset, overlapLength)
          } else {
            Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
              "only support Float and Double")
          }
        case _ =>
      }
      i += 1
    }

    this
  }

  override def compress(tensor: Tensor[T]): FP16SplitsCompressedTensor.this.type =
    compress(0, tensor, 0, tensor.nElement())

  override def deCompress(srcOffset: Int, tensor: Tensor[T], tgtOffset: Int, length: Int): Unit = {
    Log4Error.unKnowExceptionError(tensor.isContiguous(),
      "tensor is expected to be contiguous")
    Log4Error.unKnowExceptionError(srcOffset >= 0 && length > 0 &&
      srcOffset + length <= size,
      s"srcOffset $srcOffset cannot be negative, length $length should be positive," +
        s" srcOffset $srcOffset + length $length should not greater than size ${size}")
    Log4Error.unKnowExceptionError(tgtOffset >= 0 && tgtOffset + length <= tensor.nElement(),
      s"tgtOffset $tgtOffset cannot be negative," +
        s"tgtOffset $tgtOffset + length $length should not greater than " +
        s"tensor.nElement() ${tensor.nElement()}")
    val splitSize = size / buffers.length
    val extraSize = size % buffers.length
    var i = 0
    while (i < buffers.length) {
      val start = splitSize * i + math.min(extraSize, i)
      val curLength = splitSize + (if (i < extraSize) 1 else 0)
      overlap(start, curLength, srcOffset, length) match {
        case Some((splitOffset, overlapLength)) =>
          if (classTag[T] == classTag[Double]) {
            val tdata = tensor.storage().array().asInstanceOf[Array[Double]]
            val toffset = tensor.storageOffset() - 1 + tgtOffset
            FP16CompressedTensor.fromFP16(buffers(i), splitOffset * 2, overlapLength * 2,
              tdata, toffset + start)
          } else if (classTag[T] == classTag[Float]) {
            val tdata = tensor.storage().array().asInstanceOf[Array[Float]]
            val toffset = tensor.storageOffset() - 1 + tgtOffset
            FP16CompressedTensor.fromFP16(buffers(i), splitOffset * 2, overlapLength * 2,
              tdata, toffset + start)
          } else {
            Log4Error.invalidInputError(false, s"${classTag[T]} is not supported",
              "only support Float and Double")
          }
        case _ =>
      }
      i += 1
    }
  }

  override def deCompress(tensor: Tensor[T]): Unit = deCompress(0, tensor, 0, tensor.nElement())

  override def bytes(offset: Int, length: Int): ByteBuffer = {
    val splitSize = size / buffers.length
    val extraSize = size % buffers.length
    var i = 0
    while (i < buffers.length) {
      val start = splitSize * i + math.min(extraSize, i)
      val curLength = splitSize + (if (i < extraSize) 1 else 0)
      if (start == offset && curLength == length) {
        Log4Error.unKnowExceptionError(buffers(i) != null, "split has not been inited")
        return ByteBuffer.wrap(buffers(i))
      }
      i += 1
    }
    Log4Error.invalidOperationError(false, "Offset and length not match")
    null
  }

  override def bytes(): ByteBuffer = bytes(0, size)

  // scalastyle:off
  override def add(data: ByteBuffer, offset: Int,
    length: Int): FP16SplitsCompressedTensor.this.type = ???

  override def add(data: ByteBuffer): FP16SplitsCompressedTensor.this.type = ???

  override def parAdd(data: ByteBuffer, offset: Int,
    length: Int): FP16SplitsCompressedTensor.this.type = ???

  override def parAdd(data: ByteBuffer): FP16SplitsCompressedTensor.this.type = ???
  // scalastyle:on
}
