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

import com.intel.analytics.bigdl.utils.{TestUtils, T}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

@com.intel.analytics.bigdl.tags.Parallel
class LBFGSSpec extends FlatSpec with Matchers {
  "torchLBFGS in regular batch test" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val optm = new LBFGS[Double]
    val result = optm.optimize(TestUtils.rosenBrock, x,
      T("maxIter" -> 100, "learningRate" -> 1e-1))
    val fx = result._2

    println()
    println("Rosenbrock test")
    println()

    println(s"x = $x")
    println("fx = ")
    for (i <- 1 to fx.length) {
      println(s"$i ${fx(i - 1)}")
    }
    println()
    println()

    fx.last < 1e-6 should be(true)
  }

  "torchLBFGS in stochastic test" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val optm = new LBFGS[Double]
    val fx = new ArrayBuffer[Double]()

    val config = T("maxIter" -> 1, "learningRate" -> 1e-1)
    for (i <- 1 to 100) {
      val result = optm.optimize(TestUtils.rosenBrock, x, config)
      fx.append(result._2(0))
    }

    println()
    println("Rosenbrock test")
    println()

    println(s"x = $x")
    println("fx = ")
    for (i <- 1 to fx.length) {
      println(s"$i ${fx(i - 1)}")
    }
    println()
    println()

    fx.last < 1e-6 should be(true)
  }
}
