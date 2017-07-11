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

import com.intel.analytics.bigdl.fixpoint.FixPoint
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

@SerialVersionUID(- 1766499387282335147L)
private[bigdl] class QuantizeTensor[@specialized(Float) T: ClassTag](val tensorType: Int)
  (implicit ev: TensorNumeric[T]) extends Serializable {
  private var realSize = 0L

  @transient private var desc = 0L
  private var interStorage: Option[Array[Byte]] = None

  def setBufferFromInterStorage(buffer: ByteBuffer): Unit = {
    interStorage = Some(buffer.array())
  }

  def getBufferFromInterStorage: Option[Array[Byte]] = {
    interStorage
  }

  def setStorageInJni(ptr: Long): Unit = {
    desc = ptr
  }

  def getStorageInJni: Long = {
    desc
  }

  def isInitialized: Boolean = {
    if (desc == 0) {
      false
    } else {
      true
    }
  }

  def release(): Unit = {
    FixPoint.FreeMemory(desc, tensorType)
  }
}

object QuantizeTensor {
  def apply[@specialized(Float, Double) T: ClassTag](tensorType: Int)(
    implicit ev: TensorNumeric[T]): QuantizeTensor[T] = new QuantizeTensor[T](tensorType)
}
