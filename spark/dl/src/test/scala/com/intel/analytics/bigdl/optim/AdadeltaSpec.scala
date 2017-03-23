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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, TestUtils}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

@com.intel.analytics.bigdl.tags.Parallel
class AdadeltaSpec extends FlatSpec with Matchers {
  val start = System.currentTimeMillis()
  "adadelta" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val config = T("Epsilon" -> 1e-10)
    val optm = new Adadelta[Double]
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

    (fx.last < 1e-4) should be(true)
    x(Array(1)) should be(1.0 +- 0.02)
    x(Array(2)) should be(1.0 +- 0.02)
  }
}

