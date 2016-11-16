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

import com.intel.analytics.sparkdl.nn.JoinTable
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{RandomGenerator, T, Table}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable

class JoinTableSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A JoinTable()" should "generate correct output and grad" in {
    def randomn(): Double = RandomGenerator.RNG.uniform(-10, 10)
    val layer = new JoinTable[Double](2, 2)

    val input1 = Tensor[Double](3, 3, 3)
    input1.apply1(x => randomn())
    val input2 = Tensor[Double](3, 3, 3)
    input2.apply1(x => randomn())
    val input = T()
    input(1.toDouble) = input1
    input(2.toDouble) = input2
    val gradOutput = Tensor[Double](3, 3, 6)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.JoinTable(2, 2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[mutable.HashMap[Any, Any]]

    output should be (luaOutput)
    gradInput should be (new Table(luaGradInput))

    println("Test case : JoinTable, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
