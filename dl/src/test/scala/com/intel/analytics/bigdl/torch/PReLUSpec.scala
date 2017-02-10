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

import com.intel.analytics.bigdl.nn.PReLU
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class PReLUSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A PReLU Module " should "generate correct output and grad not inplace" in {
    val module = new PReLU[Double]()
    val input = Tensor[Double](2, 3, 4).apply1(_ => Random.nextDouble() - 0.5)
    val gradOutput = Tensor[Double](2, 3, 4).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.PReLU()\n" +
      "module:zeroGradParameters()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n" +
      "gradWeight = module.gradWeight"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradWeight"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    luaOutput should be (output)
    luaGradInput should be (gradInput)
    luaGradWeight should be (module.gradWeight)

    println("Test case : PReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A PReLU(2)" should "generate correct output and grad not inplace" in {
    val module = new PReLU[Double](2)
    val input = Tensor[Double](2, 3, 4).apply1(_ => Random.nextDouble() - 0.5)
    val gradOutput = Tensor[Double](2, 3, 4).apply1(_ => Random.nextDouble())

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.PReLU(2)\n" +
      "module:zeroGradParameters()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n" +
      "gradWeight = module.gradWeight"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "gradWeight"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaGradWeight = torchResult("gradWeight").asInstanceOf[Tensor[Double]]

    luaOutput should be (output)
    luaGradInput should be (gradInput)
    luaGradWeight should be (module.gradWeight)

    println("Test case : PReLU, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
