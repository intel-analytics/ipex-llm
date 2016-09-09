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

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.tensor.{Tensor, torch}
import com.intel.analytics.sparkdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

class SGDSpec extends FlatSpec with Matchers {
  "A SGD optimMethod with 1 parameter" should "generate correct result" in {
    val state = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.01, "momentum" -> 0.002)
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val r = x.clone()
      r.apply1(2 * _)
      val v = x(Array(1))
      return (v * v, r)
    }
    val x = torch.Tensor[Double](1)
    x.fill(10)
    for (i <- 1 to 10) {
      optimMethod.optimize(feval, x, state, state)
    }
    x(Array(1)) should be(1.0591906190415 +- 1e-6)
  }

  "sgd" should "perform well on rosenbrock function" in {

    val x = torch.Tensor[Double](2).fill(0)
    val config = T("learningRate" -> 1e-3)
    val optm = new SGD[Double]
    var fx = new ArrayBuffer[Double]
    for (i <- 1 to 10001) {
      val result = optm.optimize(TestUtils.rosenBrock, x, config)
      if ((i - 1) % 1000 == 0) {
        fx += (result._2(0))
      }
    }

    println(s"x is \n$x")
    println("fx is")
    for (i <- 1 to fx.length) {
      println(s"${(i - 1) * 1000 + 1}, ${fx(i - 1)}")
    }

    (fx.last < 1e-4) should be(true)
    x(Array(1)) should be(1.0 +- 0.1)
    x(Array(2)) should be(1.0 +- 0.1)
  }
}
