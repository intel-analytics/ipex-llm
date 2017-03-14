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

import com.intel.analytics.bigdl.nn.LookupTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class LookupTableSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A LookupTableSpec" should "generate correct output and grad with input 1D" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](9, 4, 2, 0.1, 2.0, true)
    val input = Tensor[Double](5)
    input(Array(1)) = 5
    input(Array(2)) = 2
    input(Array(3)) = 6
    input(Array(4)) = 9
    input(Array(5)) = 4

    val gradOutput = Tensor[Double](2, 2, 2)

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LookupTable(9, 4, 2, 0.1)\n" +
      "module:scaleGradByFreq()\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input:int())\n" +
      "_gradInput = module:backward(input:int(), output)\n" +
      "i = i + 1\n" +
      "end\n" +
      "gradInput = _gradInput:double()\n" +
      "weight = module.weight\n" +
      "gradweight = module.gradWeight\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "weight", "gradweight"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaweight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradweight").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    var output : Tensor[Double] = null
    var gradInput : Tensor[Double] = null
    var i = 0
    while (i < 10) {
      output = module.forward(input)
      gradInput = module.backward(input, output)
      i += 1
    }
    val weight = module.weight
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    luaweight should be(weight)
    luagradWeight should be (gradWeight)
    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : LookupTable, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A LookupTableSpec" should "generate correct output and grad with input 2D" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](10, 3, 3)
    val input = Tensor[Double](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 10

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LookupTable(10, 3, 3)\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input)\n" +
      "_gradInput = module:backward(input, output)\n" +
      "i = i + 1\n" +
      "end\n" +
      "gradInput = _gradInput:double()\n" +
      "weight = module.weight\n" +
      "gradweight = module.gradWeight\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput", "weight", "gradweight", "shouldScaleGradByFreq"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaweight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradweight").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    var output : Tensor[Double] = null
    var gradInput : Tensor[Double] = null
    var i = 0
    while (i < 10) {
      output = module.updateOutput(input)
      gradInput = module.backward(input, output)
      i += 1
    }
    val weight = module.weight
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    luaweight should be(weight)
    luagradWeight should be (gradWeight)
    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : LookupTable, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A LookupTableSpec" should "generate correct output and grad with max-norm regularization" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](10, 3, 0, 0.1, 2)
    val input = Tensor[Double](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 10

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.LookupTable(10, 3, 0, 0.1, 2)\n" +
      "local i = 0\n" +
      "while i < 10 do\n" +
      "output = module:forward(input:int())\n" +
      "_gradInput = module:backward(input:int(),output) \n" +
      "i = i + 1\n" +
      "end\n" +
      "gradInput = _gradInput:double()\n" +
      "weight = module.weight\n" +
      "gradweight = module.gradWeight\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("output", "gradInput", "weight", "gradweight"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]
    val luaweight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luagradWeight = torchResult("gradweight").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    var output : Tensor[Double] = null
    var gradInput : Tensor[Double] = null
    var i = 0
    while (i < 10) {
      output = module.forward(input)
      gradInput = module.backward(input, output)
      i += 1
    }
    val weight = module.weight
    val gradWeight = module.gradWeight
    val end = System.nanoTime()
    val scalaTime = end - start

    luaweight should be(weight)
    luagradWeight should be (gradWeight)
    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)

    println("Test case : LookupTable, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }
}
