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

import com.intel.analytics.bigdl.nn.LogSigmoid
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class LogSigmoidSpec extends TorchSpec {
    "A LogSigmoid Module " should "generate correct output and grad" in {
    torchCheck()
    val module = new LogSigmoid[Double]()
    Random.setSeed(100)
    val input = Tensor[Double](4, 10).apply1(e => Random.nextDouble())
    val data = Tensor[Double](4, 20).randn()
    val gradOutput = data.narrow(2, 1, 10)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.LogSigmoid()\n" +
      "output1 = module:forward(input)\n " +
      "output2 = module:backward(input, gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output1", "output2"))
    val luaOutput = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput should be(output)
    luaGradInput should be(gradInput)

    println("Test case : LogSigmoid, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
