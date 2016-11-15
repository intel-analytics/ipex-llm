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

import com.intel.analytics.sparkdl.nn.Index
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap
import scala.util.Random

class IndexSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A Index " should "generate correct output and grad with one dimension" in {
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](3)
    input1(Array(1)) = 10
    input1(Array(2)) = 20
    input1(Array(3)) = 30

    val input2 = Tensor[Double](4)
    input2(Array(1)) = 1
    input2(Array(2)) = 2
    input2(Array(3)) = 2
    input2(Array(4)) = 3
    val gradOutput = Tensor[Double](4).apply1(e => Random.nextDouble())

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val code = "torch.manualSeed(" + seed + ")\n" +
      "input = {input1, torch.LongTensor{1, 2, 2, 3}}\n" +
      "module = nn.Index(1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    val module = new Index[Double](1)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)

    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null)

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    gradInput2 should be(luagradInput2)

    println("Test case : Index, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Index " should "generate correct output and grad with two dimension" in {
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](3, 3)
    input1(Array(1, 1)) = 10
    input1(Array(1, 2)) = 20
    input1(Array(1, 3)) = 30
    input1(Array(2, 1)) = 100
    input1(Array(2, 2)) = 200
    input1(Array(2, 3)) = 300
    input1(Array(3, 1)) = 1
    input1(Array(3, 2)) = 2
    input1(Array(3, 3)) = 3

    val input2 = Tensor[Double](4)
    input2(Array(1)) = 1
    input2(Array(2)) = 2
    input2(Array(3)) = 3
    input2(Array(4)) = 1

    val gradOutput = Tensor[Double](3, 4).apply1(e => Random.nextDouble())

    val input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val code = "torch.manualSeed(" + seed + ")\n" +
      "input = {input1, torch.LongTensor{1, 2, 3, 1}}\n" +
      "module = nn.Index(2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    val module = new Index[Double](2)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)

    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null)

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    gradInput2 should be(luagradInput2)

    println("Test case : Index, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }


}
