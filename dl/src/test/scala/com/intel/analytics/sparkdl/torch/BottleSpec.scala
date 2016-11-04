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

import com.intel.analytics.sparkdl.nn.{Bottle, Linear}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class BottleSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A Bottle Container" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = new Bottle[Tensor[Double], Tensor[Double], Double](new Linear(10, 2), 2, 2)
    module.add(new Linear(10, 2))

    var tmp = Tensor[Double](3)
    tmp.setValue(1,1)
    tmp(1) = 1
    tmp(2) = 2
    tmp(3) = 3

    val input = Tensor[Double](4, 5, 10).fill(RNG.uniform(-1, 1)) //fill(1)//.
    val gradOutput = Tensor[Double](4, 10).randn()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "torch.manualSeed(" + seed + ")\n" +
    "module = nn.Bottle(nn.Linear(10,2), 2, 2)\n" +
    "inShape = module.inShape\n" +
    "outShape = module.outShape\n" +
    "output = module:forward(input)\n" +
    "gradInput = module:backward(input,gradOutput)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput","inShape","outShape"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output)
    luaOutput2 should be(gradInput)
    println("Test case : Sequential, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
