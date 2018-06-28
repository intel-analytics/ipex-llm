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

import java.util

import com.intel.analytics.bigdl.mkl.Memory

import scala.reflect._

private[tensor] class ArrayStorage[@specialized(Double, Float) T: ClassTag](
  private[tensor] var values: Array[T]) extends Storage[T] {

  override def apply(index: Int): T = values(index)

  override def update(index: Int, value: T): Unit = values(index) = value

  override def length(): Int = values.length

  override def iterator: Iterator[T] = values.iterator

  override def array(): Array[T] = values

  override def copy(source: Storage[T], offset: Int, sourceOffset: Int,
    length: Int): this.type = {
    source match {
      case s: ArrayStorage[T] => System.arraycopy(s.values, sourceOffset,
        this.values, offset, length)
      case s: DnnStorage[T] =>
        require(classTag[T] == ClassTag.Float, "Only support copy float dnn storage")
        require(sourceOffset == 0, "dnn storage offset should be 0")
        Memory.CopyPtr2Array(s.ptr.address, 0, values.asInstanceOf[Array[Float]], offset, length,
          DnnStorage.FLOAT_BYTES)
      case _ => throw new UnsupportedOperationException("Only support dnn or array storage")
    }
    this
  }

  override def resize(size: Long): this.type = {
    values = new Array[T](size.toInt)
    this
  }

  override def fill(value: T, offset: Int, length: Int): this.type = {

    value match {
      case v: Double => util.Arrays.fill(values.asInstanceOf[Array[Double]],
        offset - 1, offset - 1 + length, v)
      case v: Float => util.Arrays.fill(values.asInstanceOf[Array[Float]],
        offset - 1, offset - 1 + length, v)
      case v: Int => util.Arrays.fill(values.asInstanceOf[Array[Int]],
        offset - 1, offset - 1 + length, v)
      case v: Long => util.Arrays.fill(values.asInstanceOf[Array[Long]],
        offset - 1, offset - 1 + length, v)
      case v: Short => util.Arrays.fill(values.asInstanceOf[Array[Short]],
        offset - 1, offset - 1 + length, v)
      case _ => throw new IllegalArgumentException
    }

    this
  }

  override def set(other: Storage[T]): this.type = {
    require(other.length() == this.length())
    this.values = other.array
    this
  }
}
