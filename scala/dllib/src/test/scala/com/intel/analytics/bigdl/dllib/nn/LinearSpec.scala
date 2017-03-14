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

package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl._

import scala.math._
import com.intel.analytics.bigdl._

@com.intel.analytics.bigdl.tags.Parallel
class LinearSpec extends FlatSpec with Matchers {
  "Linear module" should "converge to correct weight and bias" in {
    val inputN = 5
    val outputN = 2

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](inputN)
    val res = Tensor[Double](outputN)
    var err = 0.0
    for (i <- 1 to 10000) {
      input.rand()
      for (y <- 1 to outputN) {
        res(Array(y)) = 1.0 * y
        for (x <- 1 to inputN) {
          res(Array(y)) += 0.1 * y * x * input(Array(x))
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input, grad)
      linear.updateParameters(0.5 / log(i + 3))
    }
    val params = linear.parameters()
    val weight = params._1(0)
    val bias = params._1(1)

    val expectedWeight = Tensor[Double](outputN, inputN)
    val expectedBias = Tensor[Double](outputN)
    for (y <- 1 to outputN) {
      expectedBias(Array(y)) = 1.0 * y
      for (x <- 1 to inputN) {
        expectedWeight(Array(y, x)) = 0.1 * y * x
      }
    }

    expectedBias.map(bias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedWeight.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(err < 1e-6)
  }

  "Linear module in batch mode" should "converge to correct weight and bias" in {
    val inputN = 5
    val outputN = 2
    val batchN = 3

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](batchN, inputN)
    val res = Tensor[Double](batchN, outputN)
    var err = 0.0
    for (i <- 1 to 10000) {
      input.rand()
      for (k <- 1 to batchN) {
        for (y <- 1 to outputN) {
          res(Array(k, y)) = 1.0 * y
          for (x <- 1 to inputN) {
            res(Array(k, y)) += 0.1 * y * x * input(Array(k, x))
          }
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input, grad)
      linear.updateParameters(0.5 / log(i + 3))
    }
    val params = linear.parameters()
    val weight = params._1(0)
    val bias = params._1(1)

    val expectedWeight = Tensor[Double](outputN, inputN)
    val expectedBias = Tensor[Double](outputN)
    for (y <- 1 to outputN) {
      expectedBias(Array(y)) = 1.0 * y
      for (x <- 1 to inputN) {
        expectedWeight(Array(y, x)) = 0.1 * y * x
      }
    }

    expectedBias.map(bias, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedWeight.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(err < 1e-6)
  }

  "Linear module in batch mode without bias" should "converate to correct weight and bias" in {
    val inputN = 5
    val outputN = 2
    val batchN = 3

    val linear = new Linear[Double](inputN, outputN, withBias = false)
    val mse = new MSECriterion[Double]

    val input = Tensor[Double](batchN, inputN)
    val res = Tensor[Double](batchN, outputN)
    var err = 0.0
    for (i <- 1 to 10000) {
      input.rand()
      for (k <- 1 to batchN) {
        for (y <- 1 to outputN) {
          res(Array(k, y)) = 0
          for (x <- 1 to inputN) {
            res(Array(k, y)) += 0.1 * y * x * input(Array(k, x))
          }
        }
      }
      val output = linear.forward(input)
      err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.zeroGradParameters()
      linear.backward(input, grad)
      linear.updateParameters(0.5 / log(i + 3))
    }
    val params = linear.parameters()
    val weight = params._1(0)

    val expectedWeight = Tensor[Double](outputN, inputN)
    for (y <- 1 to outputN) {
      for (x <- 1 to inputN) {
        expectedWeight(Array(y, x)) = 0.1 * y * x
      }
    }

    expectedWeight.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(err < 1e-6)
  }

  "Linear module in batch mode" should "be good in gradient check" in {
    val linear = new Linear[Double](5, 2)
    linear.reset()
    val input = Tensor[Double](3, 5).rand()

    val checker = new GradientChecker(1e-4, 1e-2)
    checker.checkLayer[Double](linear, input) should be(true)
  }
}
