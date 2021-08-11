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

import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.tensor.Tensor

@com.intel.analytics.bigdl.tags.Serial
class ClassNLLCriterionSpec extends TorchSpec {
  "A ClassNLL Criterion " should "generate correct output and grad" in {
    torchCheck()
    val criterion = new ClassNLLCriterion[Double]()
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.0262627674932
    input(Array(1, 2)) = -1.2412600935171
    input(Array(1, 3)) = -1.0423174168648
    input(Array(2, 1)) = -0.90330565804228
    input(Array(2, 2)) = -1.3686840144413
    input(Array(2, 3)) = -1.0778380454479
    input(Array(3, 1)) = -0.99131220658219
    input(Array(3, 2)) = -1.0559142847536
    input(Array(3, 3)) = -1.2692712660404
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "criterion = nn.ClassNLLCriterion()\n" +
      "output1 = criterion:forward(input, target)\n " +
      "output2 = criterion:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : ClassNLLCriterion, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }
}
