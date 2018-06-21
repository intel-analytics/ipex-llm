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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}

class JoinTableSpec extends BigDLSpecHelper {
  "Join table" should "work correctly" in {
    val layer = new JoinTable(1)
    val model = Sequential()
    val concat = ConcatTable()
    concat.add(ReorderMemory(NativeData(Array(2, 2), Memory.Format.nc)))
    concat.add(ReorderMemory(NativeData(Array(2, 2), Memory.Format.nc)))
    val input = Input(Array(2, 2), Memory.Format.nc)
    model.add(input)
    model.add(concat)
    model.add(layer)
    model.add(ReorderMemory(HeapData(Array(4, 2), Memory.Format.nc)))
    model.compile(Phase.TrainingPhase)
    model.forward(Tensor[Float](T(T(1, 2), T(3, 4)))) should be(Tensor[Float](T(
      T(1, 2),
      T(3, 4),
      T(1, 2),
      T(3, 4)
    )))
  }
}
