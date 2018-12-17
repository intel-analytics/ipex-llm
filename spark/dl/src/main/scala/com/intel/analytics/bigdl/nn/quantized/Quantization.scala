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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.quantized.Utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import java.nio.ByteBuffer
import scala.reflect.ClassTag

object Quantization {
  def findMax(src: Array[Float], start: Int, end: Int): Float = {
    src.slice(start, end).max
  }

  def findMin(src: Array[Float], start: Int, end: Int): Float = {
    src.slice(start, end).min
  }

  def quantize(value: Float, max: Float, min: Float): Byte = {
    Math.round(1.0 * value / Math.max(Math.abs(max), Math.abs(min)) * Byte.MaxValue).toByte
  }

  def dequantize(byte: Byte, max: Float, min: Float): Float = {
    byte.toFloat / Byte.MaxValue * Math.max(Math.abs(max), Math.abs(min))
  }

  def quantize(src: Array[Float], start: Int, end: Int, dst: Array[Byte],
    dstOffset: Int): (Float, Float) = {
    // we should keep the plus and minus
    val max = findMax(src, start, end)
    val min = findMin(src, start, end)

    for (i <- 0 until end - start) {
      dst(dstOffset + i) = quantize(src(start + i), max, min)
    }

    (max, min)
  }

  def dequantize(src: Array[Float], start: Int, end: Int, dst: Array[Byte], dstOffset: Int,
    max: Float, min: Float): Unit = {
    require(src.length >= end, s"you write too much elements")

    for (i <- 0 until end - start) {
      src(start + i) = dequantize(dst(dstOffset + i), max, min)
    }
  }

  def quantize(src: Array[Float], start: Int, end: Int, dst: Array[Byte], dstOffset: Int,
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

  def dequantize(data: Array[Float], start: Int, end: Int, quantizedData: Array[Byte], offset: Int,
    max: Array[Float], min: Array[Float], size: Array[Int]): Unit = {
    require(max.length == min.length, s"the number of max doesn't match with the number of min")
    require(size.length == 2, s"only support 2-dim matrix")
    require(max.length == size(0),
      s"the number of max(${max.length}) doesn't match the size(${size(1)})")

    require(size.product == (end - start), s"number of elements does not match")

    val height = size(0)
    val width = size(1)

    for (i <- 0 until height) {
      dequantize(data, start + i * width, start + (i + 1) * width,
        quantizedData, offset + i * width, max(i), min(i))
    }
  }

  private[bigdl] def get2Dim(shape: Array[Int]): Array[Int] = {
    require(shape.length > 1, s"error size dimension, which must be great than 1")
    val first = shape(0)
    val last = shape.slice(1, shape.length).product
    Array(first, last)
  }

  def quantize(input: Tensor[Float], buffer: Array[Byte],
    offset: Int): (Array[Float], Array[Float]) = {
    val length = input.nElement()

    input.dim() match {
      case 1 =>
        val (max, min) = quantize(input.storage().array(), input.storageOffset() - 1,
          length, buffer, offset)
        (Array(max), Array(min))
      case x if x > 1 =>
        val size = get2Dim(input.size())
        val start = input.storageOffset() - 1
        val end = start + length
        val (max, min) = quantize(input.storage().array(), start, end, buffer, offset, size)
        (max, min)
      case _ => throw new UnsupportedOperationException(s"unsupported input")
    }
  }

  def dequantize(input: Tensor[Float], buffer: Array[Byte], offset: Int, max: Array[Float],
    min: Array[Float]): Unit = {
    val start = input.storageOffset() - 1
    val end = start + input.nElement()

    input.dim() match {
      case 1 => dequantize(input.storage().array(), start, end, buffer,
        offset, max(0), min(0))
      case x if x > 1 =>
        dequantize(input.storage().array(), start, end, buffer,
          offset, max, min, get2Dim(input.size()))
      case _ => throw new UnsupportedOperationException {
        s"unsupported input dim ${input.dim()}"
      }
    }
  }

  def loss(before: Array[Float], after: Array[Float], start: Int, end: Int): Double = {
    var lossValue = 0.0

    for (i <- start until end) {
      lossValue += Math.abs(before(i) - after(i))
    }

    lossValue
  }

  def loss(before: Tensor[Float], after: Tensor[Float]): Double = {
    val beforeArray = before.storage().array()
    val afterArray = after.storage().array()

    val start = 0
    val end = before.nElement()

    loss(beforeArray, afterArray, start, end) / beforeArray.sum
  }

  def quantize[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    // deep copy a new model then substitute with all quantized version modules
    val clonedModel = model.cloneModule()
    println("Converting model now")
    val quantizedModel = Quantizer.quantize(clonedModel)
    println("Converting model successfully")

    val paras = quantizedModel.parameters()._1
    reorganizeParameters(paras)

    quantizedModel
  }
}
