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

import com.intel.analytics.bigdl.nn.Threshold
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine

import scala.math._

@com.intel.analytics.bigdl.tags.Serial
class ThresholdSpec extends TorchSpec {
    "A Threshold Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new Threshold[Double](1, 0.8)
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.89699813351035
    input(Array(1, 1, 2)) = 1.8529373928905
    input(Array(1, 2, 1)) = 1.8799053365365
    input(Array(1, 2, 2)) = 0.076761466450989
    input(Array(2, 1, 1)) = 1.8863626234233
    input(Array(2, 1, 2)) = 0.73405137099326
    input(Array(2, 2, 1)) = 1.3404842875898
    input(Array(2, 2, 2)) = -0.64910735283047
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.31924905977212
    gradOutput(Array(1, 1, 2)) = 0.22160539613105
    gradOutput(Array(1, 2, 1)) = 0.19705923949368
    gradOutput(Array(1, 2, 2)) = 0.386440459406
    gradOutput(Array(2, 1, 1)) = 0.12920403806493
    gradOutput(Array(2, 1, 2)) = 0.7669838971924
    gradOutput(Array(2, 2, 1)) = 0.10939974407665
    gradOutput(Array(2, 2, 2)) = 0.70845287665725

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("module" -> module, "input" -> input,
      "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

    println("Test case : Threshold, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
