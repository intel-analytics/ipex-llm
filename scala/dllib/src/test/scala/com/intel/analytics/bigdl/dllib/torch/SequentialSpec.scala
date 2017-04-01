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

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._

@com.intel.analytics.bigdl.tags.Serial
class SequentialSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A Sequential Container" should "generate correct output and grad" in {
    val module = new Sequential[Double]()
    module.add(new Linear(10, 25))
    module.add(new Linear(25, 10))

    val input = Tensor[Double](10).randn()
    val gradOutput = Tensor[Double](10).randn()

    val start = System.nanoTime()
    val output = module.forward(input).toTensor[Double]
    val gradInput = module.backward(input, gradOutput).toTensor[Double]
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

    println("Test case : Sequential, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Sequential Container" should "update weight correctly" in {
    val module = new Sequential[Double]()
    module.add(new Linear(10, 25))
    module.add(new Linear(25, 10))

    val input = Tensor[Double](10).randn()
    val gradOutput = Tensor[Double](10).randn()

    val start = System.nanoTime()
    val (weight, grad) = module.getParameters()
    var i = 0
    var output = Tensor[Double]()
    var gradInput = Tensor[Double]()
    while (i < 10) {
      output = module.forward(input).toTensor[Double]
      gradInput = module.updateGradInput(input, gradOutput).toTensor[Double]
      module.accGradParameters(input, gradOutput)
      i += 1
    }
    val end = System.nanoTime()
    val scalaTime = end - start

    val code =
      "local i = 0\n" +
        "while i < 10 do\n" +
        "output = module:forward(input)\n" +
        "gradInput = module:backward(input,gradOutput)\n" +
        "i = i + 1\n" +
        "end"

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

    println("Test case : Sequential, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
