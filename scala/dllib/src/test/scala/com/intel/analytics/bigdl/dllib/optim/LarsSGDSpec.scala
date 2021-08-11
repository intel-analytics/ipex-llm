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


import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.{Linear, MSECriterion, ReLU, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T, TestUtils}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.optim._

import scala.collection.mutable.ArrayBuffer

@com.intel.analytics.bigdl.tags.Parallel
class LarsSGDSpec extends FlatSpec with Matchers with BeforeAndAfter {

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
  "lars" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val optm = new LarsSGD[Double](true, _learningRate = 0.1, _learningRateDecay = 0.09)
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

    (fx.last < 1e-5) should be(true)
    x(Array(1)) should be(1.0 +- 0.01)
    x(Array(2)) should be(1.0 +- 0.01)
  }

  "lars" should "perform well on multi-layer network" in {
    def thefunction(x1: Float, x2: Float): Float = {
      x1 * x2 + Math.sin(x1).toFloat + Math.sin(x2).toFloat + Math.pow(x1, x2).toFloat
    }

    val numSamples = 200000 / 64

    def generateData(): DataSet[MiniBatch[Float]] = {
      val features = Array.tabulate(numSamples) { i => {
        val v = Tensor[Float](Array(32, 2)).rand()
        val wrapped_tag = Tensor[Float](Array(32, 1))
        (1 to 32).map(i =>
          wrapped_tag.setValue(i, 1, thefunction(v.valueAt(i, 1), v.valueAt(i, 2)))
        )
        MiniBatch(v, wrapped_tag)
      }
      }
      DataSet.array(features)
    }

    val module = Sequential[Float]()
    module.add(Linear(2, 25))
    module.add(ReLU())
    module.add(Linear(25, 25))
    module.add(ReLU())
    module.add(Linear(25, 25))
    module.add(ReLU())
    module.add(Linear(25, 25))
    module.add(ReLU())
    module.add(Linear(25, 1))
    val optimizer = Optimizer(module, generateData(), MSECriterion[Float]())
    val epochs = 6
    optimizer
       .setOptimMethods(LarsSGD.createOptimForModule(module, learningRate = 0.0001,
         learningRateDecay = 0, momentum = 0, weightDecay = 0))
       .setEndWhen(Trigger.maxEpoch(epochs))
       .optimize()
    (1 to 10).foreach(i => {
      val tensor = Tensor[Float](2).rand()
      val v1 = tensor.valueAt(1)
      val v2 = tensor.valueAt(2)
      val realv = thefunction(v1, v2)
      val difference = realv - module.forward(tensor).toTensor[Float].value()
      val deviation = difference / realv
      println(deviation)
      deviation should be(0.0f +- 0.1f)
    })
  }

  "lars" should "be found in multi-optims" in {
    val optim = Map("1" -> new SGD[Float](), "2" -> new LarsSGD[Float](false,
      _weightDecay = 1.23))
    LarsSGD.containsLarsSGD(optim).getOrElse(0.0) should be(1.23)
  }

}

