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
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, TestUtils}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class AdamSpec extends FlatSpec with Matchers {
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
  "adam" should " work fast with MKL" in {
    RandomGenerator.RNG.setSeed(100)
    val inputSize = 500
    val hiddenSize = 500
    val outputSize = 10
    val batchSize = 10
    val model = Sequential[Float]()
      .add(Linear[Float](inputSize, hiddenSize))
    for (i <- 1 to 3) {
      model.add(Linear[Float](hiddenSize, hiddenSize))
    }
    model.add(Linear[Float](hiddenSize, outputSize))
    val criterion = CrossEntropyCriterion[Float]()

    val input = Tensor[Float](batchSize, inputSize).rand
    val label = Tensor[Float](batchSize).zero
    for (i <- 1 to batchSize) {
      val nextLabel = Random.nextInt(outputSize) + 1
      label.setValue(i, nextLabel)
    }

    val (weights, grad) = model.getParameters()

    val state = T("learningRate" -> 1e-1, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)

    val adam = new Adam[Float]

    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      model.forward(input)
      criterion.forward(model.output.asInstanceOf[Tensor[Float]], label)
      model.zeroGradParameters()
      val gradOutputTest = criterion.backward(model.output.asInstanceOf[Tensor[Float]], label)
      model.backward(input, gradOutputTest)
      (criterion.output, grad)
    }

    val warmUp = 30
    val iter = 50
    for (i <- 1 to warmUp) {
      adam.optimize(feval, weights, state)
    }
    var startTime = System.nanoTime
    var duration = (System.nanoTime() - startTime) / 1e9
    var sum = 0.0
    for (i <- 1 to iter) {
      startTime = System.nanoTime
      adam.optimize(feval, weights, state)
      duration = (System.nanoTime() - startTime) / 1e9
      sum += duration
      println(s"iter-${i}, eta = ${duration} seconds")
    }
    println(s"average eta = ${sum / iter} seconds")
  }
}

