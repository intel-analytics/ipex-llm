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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T, TestUtils}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class AdamSpec extends FlatSpec with Matchers with BeforeAndAfter {

  before {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("spark.master", "local[2]")
    Engine.init
  }

  after {
    System.clearProperty("bigdl.localMode")
    System.clearProperty("spark.master")
  }


  val start = System.currentTimeMillis()
  "adam" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val config = T("learningRate" -> 0.002)
    val optm = new Adam[Double]
    var fx = new ArrayBuffer[Double]
    for (i <- 1 to 10001) {
      val result = optm.optimize(TestUtils.rosenBrock, x, config)
      if ((i - 1) % 1000 == 0) {
        fx += result._2(0)
      }
    }

    println(s"x is \n$x")
    println("fx is")
    for (i <- 1 to fx.length) {
      println(s"${(i - 1) * 1000 + 1}, ${fx(i - 1)}")
    }

    val spend = System.currentTimeMillis() - start
    println("Time Cost: " + spend + "ms")

    (fx.last < 1e-9) should be(true)
    x(Array(1)) should be(1.0 +- 0.01)
    x(Array(2)) should be(1.0 +- 0.01)
  }

  "ParallelAdam" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val optm = new ParallelAdam[Double](learningRate = 0.002, parallelNum = 2)
    var fx = new ArrayBuffer[Double]
    for (i <- 1 to 10001) {
      val result = optm.optimize(TestUtils.rosenBrock, x)
      if ((i - 1) % 1000 == 0) {
        fx += result._2(0)
      }
    }

    println(s"x is \n$x")
    println("fx is")
    for (i <- 1 to fx.length) {
      println(s"${(i - 1) * 1000 + 1}, ${fx(i - 1)}")
    }

    val spend = System.currentTimeMillis() - start
    println("Time Cost: " + spend + "ms")

    (fx.last < 1e-9) should be(true)
    x(Array(1)) should be(1.0 +- 0.01)
    x(Array(2)) should be(1.0 +- 0.01)
  }

}

