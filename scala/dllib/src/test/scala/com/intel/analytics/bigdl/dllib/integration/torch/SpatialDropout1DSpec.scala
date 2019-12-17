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

package com.intel.analytics.bigdl.integration.torch

import com.intel.analytics.bigdl.nn.{SpatialDropout1D, SpatialDropout2D}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

@com.intel.analytics.bigdl.tags.Serial
class SpatialDropout1DSpec extends TorchSpec {
  "SpatialDropout1D module with continuous input" should "converge to correct weight and bias" in {
    torchCheck()
    val module = SpatialDropout1D[Double](0.7)
    val input = Tensor[Double](3, 4, 5)
    val seed = 100

    input.rand()

    val start = System.nanoTime()
    RNG.setSeed(seed)
    val output = module.forward(input)
    println(output)
    val gradInput = module.backward(input, input.clone().fill(1))
    println(gradInput)
  }

}
