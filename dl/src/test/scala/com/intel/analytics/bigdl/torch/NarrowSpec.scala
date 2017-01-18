/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.Narrow
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NarrowSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }
  "A Narrow Module " should "generate correct output and grad with length < 0" in {
    val input = Tensor[Double](9, 4, 14).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 4, 14).apply1(e => Random.nextDouble())

    val code = "module = nn.Narrow(1, 3, -3)\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input, output)\n" +
      "i = i + 1\n" +
      "end"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Narrow[Double](1, 3, -3)
    val start = System.nanoTime()
    var i = 0
    var output = Tensor[Double]()
    var gradInput = Tensor[Double]()
    while (i < 10) {
      output = module.forward(input)
      gradInput = module.backward(input, output)
      i += 1
    }
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Narrow, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Narrow Module " should "generate correct output and grad with dimension < 0" in {
    val input = Tensor[Double](3, 10, 4).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 3, 4).apply1(e => Random.nextDouble())

    val code = "module = nn.Narrow(-2, 5, -4)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Narrow[Double](-2, 5, -4)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Narrow, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
