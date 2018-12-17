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
import com.intel.analytics.bigdl.utils.BigDLSpecHelper

class InputSpec extends BigDLSpecHelper {
  "Input" should "be correct" in {
    val layer = Input(Array(2, 2), Memory.Format.nc)
    layer.setRuntime(new MklDnnRuntime())
    layer.initFwdPrimitives(Array(), Phase.TrainingPhase)
    layer.initBwdPrimitives(Array(), Phase.TrainingPhase)
    val tensor = Tensor[Float](2, 2).rand()
    val grad = Tensor[Float](2, 2).rand()
    val output = layer.forward(tensor)
    val gradInput = layer.backward(tensor, grad)
    Tools.dense(output) should be(tensor)
    Tools.dense(gradInput) should be(grad)
  }
}
