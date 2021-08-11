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
package com.intel.analytics.bigdl.nn.ops

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class ApproximateEqual[T: ClassTag](tolerance: Float)
                        (implicit ev: TensorNumeric[T]) extends Compare[T] {
  override def compareFloat(a: Float, b: Float): Boolean = math.abs(a - b) < tolerance

  override def compareDouble(a: Double, b: Double): Boolean = math.abs(a - b) < tolerance

  override def compareChar(a: Char, b: Char): Boolean = math.abs(a - b) < tolerance

  override def compareLong(a: Long, b: Long): Boolean = math.abs(a - b) < tolerance

  override def compareShort(a: Short, b: Short): Boolean = math.abs(a - b) < tolerance

  override def compareInt(a: Int, b: Int): Boolean = math.abs(a - b) < tolerance

  override def compareBoolean(a: Boolean, b: Boolean): Boolean = {
    throw new UnsupportedOperationException("Does not support ApproximateEqual on Boolean")
  }

  override def compareByteString(a: ByteString, b: ByteString): Boolean = {
    throw new UnsupportedOperationException("Does not support ApproximateEqual on ByteString")
  }
}

object ApproximateEqual {
  def apply[T: ClassTag](tolerance: Float)
     (implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new ApproximateEqual(tolerance))
}
