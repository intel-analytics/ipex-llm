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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{DataType, Memory}
import org.scalatest.{FlatSpec, Matchers}

class MemoryDataSpec extends FlatSpec with Matchers {
  "memory data hashCode comparison data" should "work correctly" in {
    val fp32 = HeapData(Array(4, 3), Memory.Format.nc, DataType.F32)
    val int8 = HeapData(Array(4, 3), Memory.Format.nc, DataType.S8)

    fp32.hashCode() == int8.hashCode() should not be (true)
  }

  "memory data hashCode comparison native" should "work correctly" in {
    val fp32 = NativeData(Array(3, 3), Memory.Format.nc, DataType.F32)
    val int8 = NativeData(Array(3, 3), Memory.Format.nc, DataType.S8)

    fp32.hashCode() == int8.hashCode() should not be (true)
  }

  "memory data hashCode comparison heap and native" should "work correctly" in {
    val heap = HeapData(Array(3, 3), Memory.Format.nc, DataType.F32)
    val native = NativeData(Array(3, 3), Memory.Format.nc, DataType.F32)

    heap.hashCode() == native.hashCode() should not be (true)
  }
}
