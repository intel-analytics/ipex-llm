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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper

class ReorderMemorySpec extends BigDLSpecHelper {
  "From heap to native" should "be correct" in {
    val layer = ReorderMemory(new HeapData(Array(3, 4), MklDnn.MemoryFormat.nc),
      new NativeData(Array(3, 4), MklDnn.MemoryFormat.nc))

    layer.initFwdPrimitives(new MklDnnRuntime(), Phase.TrainingPhase)
    layer.initBwdPrimitives(new MklDnnRuntime(), Phase.TrainingPhase)
    layer.initGradWPrimitives(new MklDnnRuntime(), Phase.TrainingPhase)
    layer.initMemory()
    val input = Tensor[Float](3, 4).rand()
    val output = layer.forward(input)
    val grad = layer.backward(input, output)
    grad should be(input)
  }

}
