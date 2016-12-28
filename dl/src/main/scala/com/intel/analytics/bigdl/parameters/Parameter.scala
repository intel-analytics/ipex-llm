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

import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

abstract trait CompressedTensor[T] extends Serializable {

  // TODO: rename tp deCompressTo
  def deCompress(srcOffset: Int, tensor: Tensor[T], tgtOffset: Int, length: Int): Unit

  def deCompress(tensor: Tensor[T]): Unit

  def bytes(offset: Int, length: Int): ByteBuffer

  def bytes(): ByteBuffer

  def add(data: ByteBuffer, offset: Int, length: Int): this.type

  def add(data: ByteBuffer): this.type

  def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type

  def parAdd(data: ByteBuffer): this.type

  // TODO: rename tp compressTo
  def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int): this.type

  def compress(tensor: Tensor[T]): this.type
}

object SerializerInstance {
  sealed trait SerializerType

  object DoubleType extends SerializerType

  object FloatType extends SerializerType

  object Fp16Type extends SerializerType

  private var pm: SerializerType = Fp16Type

  def setSerializerType(pm: SerializerType): Unit = {
    this.pm = pm
  }

  def serialize[T: ClassTag](data: Tensor[T]): CompressedTensor[T] = {
    pm match {
      case Fp16Type => new FP16CompressedTensor[T](data)
      case _ => throw new IllegalArgumentException("Unsupported parameter type")
    }
  }

  def serialize[T: ClassTag](data: ByteBuffer): CompressedTensor[T] = {
    pm match {
      case Fp16Type => new FP16CompressedTensor[T](data)
      case _ => throw new IllegalArgumentException("Unsupported parameter type")
    }
  }
}
