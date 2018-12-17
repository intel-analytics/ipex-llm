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

import com.intel.analytics.bigdl.optim.SGD._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{TestUtils, T}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

@com.intel.analytics.bigdl.tags.Parallel
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
    val x = Tensor[Double](1)
    x.fill(10)
    for (i <- 1 to 10) {
      optimMethod.optimize(feval, x, state, state)
    }
    x(Array(1)) should be(1.0591906190415 +- 1e-6)
  }

  "sgd" should "perform well on rosenbrock function" in {

    val x = Tensor[Double](2).fill(0)
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

  "default learning rate decay without table" should "generate correct learning rates" in {
    val optimMethod = new SGD[Double](0.1, 0.1)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1 / (1 + 0 * 0.1))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1 / (1 + 1 * 0.1))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1 / (1 + 2 * 0.1))
  }

  "default learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "learningRateDecay" -> 0.1, "learningRateSchedule" ->
      Default())
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 / (1 + 0 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 / (1 + 1 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 / (1 + 2 * 0.1))
  }

  it should "be used when we leave the learningRateSchedule empty" in {
    val config = T("learningRate" -> 0.1, "learningRateDecay" -> 0.1)
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 / (1 + 0 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 / (1 + 1 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 / (1 + 2 * 0.1))
  }

  "step learning rate decay without table" should "generate correct learning rates" in {
    val optimMethod = new SGD[Double](0.1)
    optimMethod.learningRateSchedule = Step(5, 0.1)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.1 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.01 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.001 +- 1e-9)
    }
  }

  "step learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "learningRateSchedule" -> Step(5, 0.1))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.1 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.01 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.001 +- 1e-9)
    }
  }

  "multistep learning rate decay with uniform step" should "work similarly with step" in {
    val config = T("learningRate" -> 0.1,
      "learningRateSchedule" -> MultiStep(Array(5, 10), 0.1))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.1 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.01 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.001 +- 1e-9)
    }
  }

  "multistep learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1,
      "learningRateSchedule" -> MultiStep(Array(10, 15), 0.1))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.1 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.1 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.01 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      config[Double]("clr") should be(-0.001 +- 1e-9)
    }
  }

  "ploy learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "learningRateSchedule" -> Poly(3, 100))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1)
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 * (1 - 1.0 / 100) * (1 - 1.0 / 100) * (1 - 1.0 / 100))
    optimMethod.optimize(feval, x, config, state)
    config[Double]("clr") should be(-0.1 * (1 - 2.0 / 100) * (1 - 2.0 / 100) * (1 - 2.0 / 100))
  }

  "ploy learning rate decay without table" should "generate correct learning rates" in {
    val optimMethod = new SGD[Double](0.1)
    optimMethod.learningRateSchedule = Poly(3, 100)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should
      be(-0.1 * (1 - 1.0 / 100) * (1 - 1.0 / 100) * (1 - 1.0 / 100))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should
      be(-0.1 * (1 - 2.0 / 100) * (1 - 2.0 / 100) * (1 - 2.0 / 100))
  }

  "epoch decay without table" should "generate correct learning rates" in {
    val regimes: Array[Regime] = Array(
      Regime(1, 3, T("learningRate" -> 1e-2, "weightDecay" -> 2e-4)),
      Regime(4, 7, T("learningRate" -> 5e-3, "weightDecay" -> 2e-4)),
      Regime(8, 10, T("learningRate" -> 1e-3, "weightDecay" -> 0.0))
    )

    val state = T("epoch" -> 0)
    val optimMethod = new SGD[Double](0.1)
    optimMethod.learningRateSchedule = EpochSchedule(regimes)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    for(e <- 1 to 10) {
      state("epoch") = e
      optimMethod.state = state
      optimMethod.optimize(feval, x)
      if(e <= 3) {
        optimMethod.learningRateSchedule.currentRate should be(-1e-2)
        optimMethod.weightDecay should be(2e-4)
      } else if (e <= 7) {
        optimMethod.learningRateSchedule.currentRate should be(-5e-3)
        optimMethod.weightDecay should be(2e-4)
      } else if (e <= 10) {
        optimMethod.learningRateSchedule.currentRate should be(-1e-3)
        optimMethod.weightDecay should be(0.0)
      }
    }
  }

  "epoch step wihout table" should "generate correct learning rates" in {

    val optimMethod = new SGD[Double](0.1)
    optimMethod.learningRateSchedule = EpochStep(1, 0.5)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T("epoch" -> 0)
    for(e <- 1 to 10) {
      state("epoch") = e
      optimMethod.state = state
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.1 * Math.pow(0.5, e))
    }
  }

  "epoch step" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "learningRateSchedule" -> EpochStep(1, 0.5))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T("epoch" -> 0)
    for(e <- 1 to 10) {
      state("epoch") = e
      optimMethod.optimize(feval, x, config, state)
      -config[Double]("clr") should be(0.1 * Math.pow(0.5, e))
    }
  }

  "Natural Exp without table" should "generate correct learning rates" in {

    val optimMethod = new SGD[Double](0.1)
    optimMethod.learningRateSchedule = NaturalExp(1, 1)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T("epoch" -> 0, "evalCounter" -> 0)
    optimMethod.state = state
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1 * math.exp(-1))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1 * math.exp(-2))
  }

  "ExponentailDecay Continous" should "generate correct learning rates" in {
    val optimMethod = new SGD[Double](0.05)
    optimMethod.learningRateSchedule = Exponential(10, 0.96)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T("epoch" -> 0, "evalCounter" -> 0)
    optimMethod.state = state
    optimMethod.optimize(feval, x)
    (1 to 5).foreach(i => {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.05 * Math.pow(0.96, i / 10.0))
    })

  }

  "ExponentailDecay Staircase" should "generate correct learning rates" in {
    val optimMethod = new SGD[Double](0.05)
    optimMethod.learningRateSchedule = Exponential(10, 0.96, true)
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T("epoch" -> 0, "evalCounter" -> 0)
    optimMethod.state = state
    (1 to 10).foreach(_ => {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.05)
    })
    (1 to 10).foreach(_ => {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.05 * Math.pow(0.96, 1))
    })
    (1 to 10).foreach(_ => {
      optimMethod.optimize(feval, x)
      optimMethod.learningRateSchedule.currentRate should be(-0.05 * Math.pow(0.96, 2))
    })
  }

  "poly learning rate decay with warmup" should "generate correct learning rates" in {
    val lrSchedules = new SequentialSchedule(100)
    lrSchedules.add(Warmup(0.3), 3).add(Poly(3, 100), 100)
    val optimMethod = new SGD[Double](learningRate = 0.1, learningRateSchedule = lrSchedules)

    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.1)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.4)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-0.7)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should be(-1.0 +- 1e-15)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should
        be(-1 * (1 - 4.0 / 100) * (1 - 4.0 / 100) * (1 - 4.0 / 100) +- 1e-15)
    optimMethod.optimize(feval, x)
    optimMethod.learningRateSchedule.currentRate should
        be(-1 * (1 - 5.0 / 100) * (1 - 5.0 / 100) * (1 - 5.0 / 100) +- 1e-15)
  }

  "ploy with warm up" should "generate correct learning rates" in {
    val optimMethod = new SGD[Double](learningRate = 0.01)
    val lrSchedule = new SequentialSchedule(10)
    lrSchedule.add(Warmup(0.01), 99).add(Poly(0.5, 1000), 1000)
    optimMethod.learningRateSchedule = lrSchedule

    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    for (i <- 0 to 1000) {
      optimMethod.optimize(feval, x)
    }
  }
}
