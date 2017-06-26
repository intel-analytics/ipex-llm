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

package com.intel.analytics.bigdl.tensor

import java.nio.ByteBuffer

object QuantizeTensor {
  def findMax(src: Array[Float], start: Int, end: Int): Float = {
    require(start < end, s"start index should great than end index")
    var max = Float.MinValue

    for (i <- start until end) {
      if (src(i) > max) max = src(i)
    }

    max
  }

  def findMin(src: Array[Float], start: Int, end: Int): Float = {
    require(start < end, s"start index should great than end index")
    var min = Float.MaxValue

    for (i <- start until end) {
      if (src(i) < min) min = src(i)
    }

    min
  }

  def quantize(value: Float, max: Float, min: Float): Byte = {
    Math.round(1.0 * (value - min) / (max - min) * Byte.MaxValue).toByte
  }

  def dequantize(byte: Byte, max: Float, min: Float): Float = {
    byte.toFloat / Byte.MaxValue * (max - min) + min
  }

  def quantize(src: Array[Float], start: Int, end: Int, dst: ByteBuffer,
    dstOffset: Int): (Float, Float) = {
    val max = Math.max(Math.abs(findMax(src, start, end)), Math.abs(findMin(src, start, end)))
    val min = findMin(src, start, end)

    for (i <- 0 until end - start) {
      dst.put(dstOffset + i, quantize(src(start + i), max, min))
    }

    (max, min)
  }

  def dequantize(src: Array[Float], start: Int, end: Int, dst: ByteBuffer, dstOffset: Int,
    max: Float, min: Float): Unit = {
    require(src.length >= end, s"you write too much elements")

    for (i <- 0 until end - start) {
      src(start + i) = dequantize(dst.get(dstOffset + i), max, min)
    }
  }

  def quantize(src: Array[Float], start: Int, end: Int, dst: ByteBuffer, dstOffset: Int,
    size: Array[Int]): (Array[Float], Array[Float]) = {
    require(size.length == 2, s"only support 2-dim matrix")
    require(size.product == (end - start), s"number of elements does not match")

    val height = size(0)
    val width = size(1)

    val max = new Array[Float](height)
    val min = new Array[Float](height)

    for (i <- 0 until height) {
      val maxAndMin = quantize(src, start + i * width, start + (i + 1) * width, dst,
        dstOffset + i * width)

      max(i) = maxAndMin._1
      min(i) = maxAndMin._2
    }

    (max, min)
  }

  def dequantize(data: Array[Float], start: Int, end: Int, quantizedData: ByteBuffer, offset: Int,
    max: Array[Float], min: Array[Float], size: Array[Int]): Unit = {
    require(max.length == min.length, s"the number of max doesn't match with the number of min")
    require(max.length == size.length, s"the number of max doesn't match the size")

    require(size.length == 2, s"only support 2-dim matrix")
    require(size.product == (end - start), s"number of elements does not match")

    val height = size(0)
    val width = size(1)

    for (i <- 0 until height) {
      dequantize(data, start + i * width, start + (i + 1) * width,
        quantizedData, offset + i * width, max(i), min(i))
    }
  }

  def loss(before: Array[Float], after: Array[Float], start: Int, end: Int): Double = {
    var lossValue = 0.0

    for (i <- start until end) {
      lossValue += (before(i) - after(i))
    }

    lossValue
  }

  def testQuantizeMatrix(): Unit = {
    val src = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
      0.6f, 0.4f, 0.3f, 0.2f, 0.1f)
    val dst = ByteBuffer.allocate(src.length)
    dst.clear()

    val (max, min) = quantize(src, 0, src.length, dst, 0, Array(2, 5))

    for (i <- src.indices) {
      println(dst.get(i))
    }

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min, Array(2, 5))
    for (i <- src.indices) {
      println(src(i))
    }
    val after = src.clone()

    println(loss(before, after, 0, src.length))
  }

  def testArray(): Unit = {
    val src = Array[Float](0.6f, 0.4f, 0.3f, 0.2f, 0.1f)

    val dst = ByteBuffer.allocate(src.length)
    dst.clear()

    quantize(src, 0, src.length, dst, 0)
    println(dst)

    for (i <- src.indices) {
      println(dst.get(i))
    }

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, 0.5f, 0.1f)
    for (i <- src.indices) {
      println(src(i))
    }

    val after = src.clone()

    println(loss(before, after, 0, src.length))
  }

  def main(args: Array[String]): Unit = {
    testArray()
    testQuantizeMatrix()
  }
}
