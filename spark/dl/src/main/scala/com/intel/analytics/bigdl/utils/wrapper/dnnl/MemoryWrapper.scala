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

package com.intel.analytics.bigdl.utils.wrapper.mkldnn

import com.intel.analytics.bigdl.dnnl.Memory

object MemoryWrapper {

  object FormatTag {
    val nc = Memory.FormatTag.nc
    val tnc = Memory.FormatTag.tnc
    val ntc = Memory.FormatTag.ntc
    val nchw = Memory.FormatTag.nchw
    val nhwc = Memory.FormatTag.nhwc
  }

  def zero(data: Long, length: Int, elementSize: Int): Long = {
    Memory.Zero(data, length, elementSize)
  }

  def copyPtr2Ptr(src: Long, srcOffset: Int, dst: Long, dstOffset: Int,
    length: Int, elementSize: Int): Long = {
      Memory.CopyPtr2Ptr(src, srcOffset, dst, dstOffset, length, elementSize)
  }

  def copyArray2Ptr(src: Array[Float], srcOffset: Int, dst: Long, dstOffset: Int,
    length: Int, elementSize: Int): Long = {
    Memory.CopyArray2Ptr(src, srcOffset, dst, dstOffset, length, elementSize)
  }

  def copyPtr2Array(src: Long, srcOffset: Int, dst: Array[Float], dstOffset: Int,
    length: Int, elementSize: Int): Long = {
    Memory.CopyPtr2Array(src, srcOffset, dst, dstOffset, length, elementSize)
  }

  def copyPtr2ByteArray(src: Long, srcOffset: Int, dst: Array[Byte], dstOffset: Int,
    length: Int, elementSize: Int): Long = {
    Memory.CopyPtr2ByteArray(src, srcOffset, dst, dstOffset, length, elementSize)
  }

  def copyPtr2IntArray(src: Long, srcOffset: Int, dst: Array[Int], dstOffset: Int,
    length: Int, elementSize: Int): Long = {
    Memory.CopyPtr2IntArray(src, srcOffset, dst, dstOffset, length, elementSize)
  }

  def sAdd(n: Int, aPtr: Long, aOffset: Int, bPtr: Long, bOffset: Int,
    yPtr: Long, yOffset: Int): Unit = {
    Memory.SAdd(n, aPtr, aOffset, bPtr, bOffset, yPtr, yOffset)
  }

  def scale(n: Int, scaleFactor: Float, from: Long, to: Long): Unit = {
    Memory.Scale(n, scaleFactor, from, to)
  }

  def axpby(n: Int, a: Float, x: Long, b: Float, y: Long): Unit = {
    Memory.Axpby(n, a, x, b, y)
  }

  def alignedMalloc(capacity: Int, size: Int): Long = {
    Memory.AlignedMalloc(capacity, size)
  }

  def alignedFree(ptr: Long): Unit = {
    Memory.AlignedFree(ptr)
  }
}
