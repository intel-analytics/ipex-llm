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

package com.intel.analytics.bigdl.dllib.integration.torch

import com.intel.analytics.bigdl.dllib.nn.SoftMax
import com.intel.analytics.bigdl.dllib.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SoftMaxSpec extends TorchSpec {
  "A SoftMax with narrowed input" should "generate correct output" in {
    val layer = new SoftMax[Double]()
    val input = Tensor[Double](4, 6).apply1(_ => Random.nextDouble())

    val in1 = input.narrow(1, 1, 2)
    val in2 = input.narrow(1, 3, 2)

    val output = layer.forward(input).clone()
    val output1 = layer.forward(in1).clone()
    val output2 = layer.forward(in2).clone()

    output.narrow(1, 1, 2) should be(output1)
    output.narrow(1, 3, 2) should be(output2)
    println("done")
  }
}
