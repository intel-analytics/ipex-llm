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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn.Masking
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.Tensor


class MaskingSpec extends KerasBaseSpec {

  "Masking" should "be ok" in {
    val batchSize = 3
    val times = 5
    val features = 2
    val inputData = Array[Double](1.0, 1, 2, 2, 3, 3, 4, 4, 5, 5, -1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
      1, 1, -1, -1, 3, 3, 4, 4, 5, 5)
    val input = Tensor[Double](inputData, Array(batchSize, times, features))

    val mask_value = -1
    val masking = Masking[Double](mask_value)

    val output = masking.forward(input)
    val gradOutput = Tensor[Double](output.size()).fill(1.0)
    val gradInput = masking.backward(input, gradOutput)

    val expectOutput = Array[Double](1.0, 1, 2, 2, 3, 3, 4, 4, 5, 5, -1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
      1, 1, 0, 0, 3, 3, 4, 4, 5, 5)
    val expectgradInput = Array[Double](1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1)
    require(output.toTensor[Double].almostEqual(Tensor[Double](expectOutput,
      Array(batchSize, times, features)), 0))
    require(gradInput.toTensor[Double].almostEqual(Tensor[Double](expectgradInput,
      Array(batchSize, times, features)), 0))
  }
}
