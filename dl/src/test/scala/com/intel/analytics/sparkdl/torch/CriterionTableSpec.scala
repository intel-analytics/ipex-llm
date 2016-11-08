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

import com.intel.analytics.sparkdl.nn.{CriterionTable, MSECriterion}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap
import scala.util.Random

class CriterionTableSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A CriterionTable " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](2, 2, 2).apply1(_ => Random.nextDouble())
    val input2 = Tensor[Double](2, 2, 2).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 2).apply1(e => Random.nextDouble())

    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2
    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CriterionTable(nn.MSECriterion())\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Double]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    val module = new CriterionTable[Double](new MSECriterion())
    val start = System.nanoTime()
    val output = module.forward(input1, input2)
    val gradInput = module.backward(input1, input2)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2.get(1).getOrElse(null))

    println("Test case : CriterionTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
