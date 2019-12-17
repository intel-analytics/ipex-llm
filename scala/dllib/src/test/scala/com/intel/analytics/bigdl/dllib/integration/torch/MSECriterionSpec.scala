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

import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.tensor.Tensor

import scala.math._

@com.intel.analytics.bigdl.tags.Serial
class MSECriterionSpec extends TorchSpec {
    "A MSE Criterion " should "generate correct output and grad" in {
    torchCheck()
    val mse = new MSECriterion[Double]
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 0.17503996845335
    input(Array(1, 1, 2)) = 0.83220188552514
    input(Array(1, 2, 1)) = 0.48450597329065
    input(Array(1, 2, 2)) = 0.64701424003579
    input(Array(2, 1, 1)) = 0.62694586534053
    input(Array(2, 1, 2)) = 0.34398410236463
    input(Array(2, 2, 1)) = 0.55356747563928
    input(Array(2, 2, 2)) = 0.20383032318205
    val target = Tensor[Double](2, 2, 2)
    target(Array(1, 1, 1)) = 0.69956525065936
    target(Array(1, 1, 2)) = 0.86074831243604
    target(Array(1, 2, 1)) = 0.54923197557218
    target(Array(1, 2, 2)) = 0.57388074393384
    target(Array(2, 1, 1)) = 0.63334444304928
    target(Array(2, 1, 2)) = 0.99680578662083
    target(Array(2, 2, 1)) = 0.49997645849362
    target(Array(2, 2, 2)) = 0.23869121982716


    val start = System.nanoTime()
    val output = mse.forward(input, target)
    val gradInput = mse.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "mse = nn.MSECriterion()\n" +
      "output = mse:forward(input,target)\n" +
      "gradInput = mse:backward(input,target)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    assert(abs(luaOutput1 - output) < 1e-6);
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
}
