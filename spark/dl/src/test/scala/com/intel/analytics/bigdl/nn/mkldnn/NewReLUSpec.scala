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

class NewReLUSpec extends BigDLSpecHelper {

  "Relu" should "be correct" in {
    val layer = ReLU(0.0f)
    val input = Tensor[Float](T(
      T(1.0, 2.0),
      T(-1.0, -2.0)
    ))
    val seq = Sequential()
    seq.add(ReorderMemory(HeapData(Array(2, 2), Memory.Format.nc),
      HeapData(Array(2, 2), Memory.Format.nc)))
    seq.add(layer)
    seq.add(ReorderMemory(HeapData(Array(2, 2), Memory.Format.nc),
      HeapData(Array(2, 2), Memory.Format.nc)))
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(2, 2), Memory.Format.nc)))
    seq.forward(input) should be(Tensor[Float](T(
      T(1.0, 2.0),
      T(0.0, 0.0)
    )))
    val grad = Tensor[Float](T(
      T(-1.0, -2.0),
      T(1.0, 2.0)
    ))
    seq.backward(input, grad) should be(Tensor[Float](T(
      T(-1.0, -2.0),
      T(0.0, 0.0)
    )))
  }

}
