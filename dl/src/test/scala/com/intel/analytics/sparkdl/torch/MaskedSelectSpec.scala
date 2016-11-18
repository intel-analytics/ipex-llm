/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.sparkdl.torch

import com.intel.analytics.sparkdl.nn.MaskedSelect
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap
import scala.util.Random

class MaskedSelectSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A MaskedSelect Module " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = new MaskedSelect[Double]()

    val input1 = Tensor[Double](2, 2).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](2, 2)
    input2(Array(1, 1)) = 1
    input2(Array(1, 2)) = 0
    input2(Array(2, 1)) = 0
    input2(Array(2, 2)) = 1

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val gradOutput = Tensor[Double](5)
    gradOutput(Array(1)) = 2
    gradOutput(Array(2)) = 5
    gradOutput(Array(3)) = 10
    gradOutput(Array(4)) = 17
    gradOutput(Array(5)) = 26

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.MaskedSelect()\n" +
      "mask = torch.ByteTensor({{1, 0}, {0, 1}})\n" +
      "output = module:forward({input1, mask})\n" +
      "gradInput = module:backward({input1, mask}, gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Any, Any]]

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be(output)

    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null)

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    // gradInput2 should be(luagradInput2)

    println("Test case : MaskedSelect, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

}
