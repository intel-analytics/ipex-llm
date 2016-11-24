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

import com.intel.analytics.sparkdl.nn.MixtureTable
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap
import scala.util.Random


class MixtureTableSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A MixtureTable " should "generate correct output and grad with table expertInput" in {
    val mse = new MixtureTable[Double]

    val expertInput = Tensor[Double](5, 3, 6).apply1(e => Random.nextDouble())
    val expertTable = new Table()
    expertTable(1.0) = Tensor[Double](5, 6).apply1(e => Random.nextDouble())//expertInput.narrow(2,1,1)
    expertTable(2.0) = Tensor[Double](5, 6).apply1(e => Random.nextDouble()) //expertInput.narrow(2,2,1)
    expertTable(3.0) = Tensor[Double](5, 6).apply1(e => Random.nextDouble()) //expertInput.narrow(2,3,1)


    val input1 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 6).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = expertTable

    val start = System.nanoTime()
    val output = mse.forward(input)
    val gradInput = mse.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "mse = nn.MixtureTable()\n" +
      "input = {input1, expertTable}\n" +
      "output = mse:forward(input)\n" +
      "gradInput = mse:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input1" -> input1, "expertTable" -> expertTable, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Any, Any]]

    output should be (luaOutput1)
    //gradInput should be (new Table(luaOutput2))

    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null).asInstanceOf[HashMap[Any, Any]]

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Table]
    gradInput2 should be(new Table(luagradInput2))

    println("Test case : MixtureTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MixtureTable " should "generate correct output and grad with tensor expertInput" in {
    val mse = new MixtureTable[Double]

    val expertInput = Tensor[Double](5, 3, 6).apply1(e => Random.nextDouble())
    val input1 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 6).apply1(e => Random.nextDouble())
    val input = new Table()
    input(1.0) = input1
    input(2.0) = expertInput

    val code = "mse = nn.MixtureTable()\n" +
      "output = mse:forward(input)\n" +
      "gradInput = mse:backward(input,gradOutput)\n" +
      "size = mse.size\n" +
      "dim = mse.dim"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput", "size", "dim"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Any, Any]]

    val start = System.nanoTime()
    val output = mse.forward(input)
    val gradInput = mse.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be (luaOutput1)
    gradInput should be (new Table(luaOutput2))

    println("Test case : MixtureTable, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
