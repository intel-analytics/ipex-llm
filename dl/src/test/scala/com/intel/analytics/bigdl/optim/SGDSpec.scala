/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.optim.SGD._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
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

  "default learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "learningRateDecay" -> 0.1, "hyperParameterScheduler" ->
      Default[Double]())
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    optimMethod.optimize(feval, x, config, state)
    var hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 / (1 + 0 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 / (1 + 1 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 / (1 + 2 * 0.1))
  }

  it should "be used when we leave the hyperParameterScheduler empty" in {
    val config = T("learningRate" -> 0.1, "learningRateDecay" -> 0.1)
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    optimMethod.optimize(feval, x, config, state)
    var hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 / (1 + 0 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 / (1 + 1 * 0.1))
    optimMethod.optimize(feval, x, config, state)
    hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 / (1 + 2 * 0.1))
  }

  "step learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "hyperParameterScheduler" -> Step[Double](5, 0.1))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      val hp = optimMethod.getHyperParameter()
      hp[Double]("learningRate") should be(0.1 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      val hp = optimMethod.getHyperParameter()
      hp[Double]("learningRate") should be(0.01 +- 1e-9)
    }

    for(i <- 1 to 5) {
      optimMethod.optimize(feval, x, config, state)
      val hp = optimMethod.getHyperParameter()
      hp[Double]("learningRate") should be(0.001 +- 1e-9)
    }
  }

  "ploy learning rate decay" should "generate correct learning rates" in {
    val config = T("learningRate" -> 0.1, "hyperParameterScheduler" -> Poly[Double](3, 100))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    optimMethod.optimize(feval, x, config, state)
    var hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1)
    optimMethod.optimize(feval, x, config, state)
    hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 * (1 - 1.0 / 100)
      * (1 - 1.0 / 100) * (1 - 1.0 / 100))
    optimMethod.optimize(feval, x, config, state)
    hp = optimMethod.getHyperParameter()
    hp[Double]("learningRate") should be(0.1 * (1 - 2.0 / 100)
      * (1 - 2.0 / 100) * (1 - 2.0 / 100))
  }

  "epoch decay" should "generate correct learning rates" in {
    val regimes: Array[Regime] = Array(
      Regime(1, 3, T("learningRate" -> 1e-2, "weightDecay" -> 2e-4)),
      Regime(4, 7, T("learningRate" -> 5e-3, "weightDecay" -> 2e-4)),
      Regime(8, 10, T("learningRate" -> 1e-3, "weightDecay" -> 0.0))
    )

    val config = T("learningRate" -> 0.1,
      "hyperParameterScheduler" -> EpochSchedule[Double](regimes))
    val optimMethod = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      return (0.1, Tensor[Double](Storage(Array(1.0, 1.0))))
    }
    val x = Tensor[Double](Storage(Array(10.0, 10.0)))
    val state = T()
    for(e <- 1 to 10) {
      state("epoch") = e
      optimMethod.optimize(feval, x, config, state)
      val hp = optimMethod.getHyperParameter()
      if(e <= 3) {
        hp[Double]("learningRate") should be(1e-2)
        hp[Double]("weightDecay") should be(2e-4)
      } else if (e <= 7) {
        hp[Double]("learningRate") should be(5e-3)
        hp[Double]("weightDecay") should be(2e-4)
      } else if (e <= 10) {
        hp[Double]("learningRate") should be(1e-3)
        hp[Double]("weightDecay") should be(0.0)
      }
    }
  }

  "getHyperParameter" should "return right hyperParameter" in {
    val config = T("learningRate" -> 1.00001,
      "learningRateDecay" -> 1e-6,
      "weightDecay" -> 1e-5,
      "momentum" -> 0.9,
      "nesterov" -> true,
      "learningRates" -> Tensor[Float](Storage(Array(1f, 2f, 3f))),
      "weightDecays" -> Tensor[Float](Storage(Array(4f, 5f, 6f))),
      "hyperParameterScheduler" -> Default[Float]())
    val state = T("evalCounter" -> 10)

    val currentHyperParameter = config[HyperParameterScheduler[Double]]("hyperParameterScheduler")
      .getAndUpdateHyperParameter(config, state)

    val clr = currentHyperParameter[Double]("learningRate")
    val lrd = currentHyperParameter[Double]("learningRateDecay")
    val wd = currentHyperParameter[Double]("weightDecay")
    val mom = currentHyperParameter[Double]("momentum")
    val damp = currentHyperParameter[Double]("dampening")
    val nesterov = currentHyperParameter[Boolean]("nesterov")
    val lrs = currentHyperParameter[Tensor[Float]]("learningRates")
    val wds = currentHyperParameter[Tensor[Float]]("weightDecays")

    clr should be (1.0)
    lrd should be (1e-6)
    wd should be (1e-5)
    mom should be (0.9)
    damp should be (mom)
    nesterov should be (true)
    lrs should be (Tensor[Float](Storage(Array(1f, 2f, 3f))))
    wds should be (Tensor[Float](Storage(Array(4f, 5f, 6f))))
  }
}
