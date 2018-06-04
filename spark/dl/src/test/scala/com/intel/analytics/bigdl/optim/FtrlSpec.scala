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
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, TestUtils}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

@com.intel.analytics.bigdl.tags.Parallel
class FtrlSpec extends FlatSpec with Matchers {
  val start = System.currentTimeMillis()
  "Ftrl" should "perform well on rosenbrock function" in {
    val x = Tensor[Double](2).fill(0)
    val config = T("learningRate" -> 0.1)
    val optm = new Ftrl[Double]
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
    x(Array(1)) should be(1.0 +- 0.01)
    x(Array(2)) should be(1.0 +- 0.01)
  }

  "Ftrl" should "works fine" in {
    val weights = Tensor[Float](2).zero()
    val grads = Tensor[Float](T(0.1f, 0.2f))
    val ftrl = new Ftrl[Float](3.0)
    (1 to 3).foreach{_ =>
      ftrl.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-2.602609f +- 0.000001f)
    weights.valueAt(2) should be (-4.296985f +- 0.000001f)
  }

  "Ftrl" should "works fine 2" in {
    val weights = Tensor[Float](2).zero()
    val grads = Tensor[Float](T(0.01f, 0.02f))
    val ftrl = new Ftrl[Float](3.0)
    (1 to 3).foreach{_ =>
      ftrl.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-0.284321f +- 0.000001f)
    weights.valueAt(2) should be (-0.566949f +- 0.000001f)
  }

  "Ftrl" should "works fine 3" in {
    val weights = Tensor[Float](T(1.0f, 2.0f))
    val grads = Tensor[Float](T(0.1f, 0.2f))
    val ftrl = new Ftrl[Float](3.0)
    (1 to 3).foreach{_ =>
      ftrl.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-2.556072f +- 0.000001f)
    weights.valueAt(2) should be (-3.987293f +- 0.000001f)
  }

  "Ftrl with L1" should "works fine 3" in {
    val weights = Tensor[Float](T(1.0f, 2.0f))
    val grads = Tensor[Float](T(0.1f, 0.2f))
    val ftrl = new Ftrl[Float](3.0, l1RegularizationStrength = 0.001)
    (1 to 10).foreach{_ =>
      ftrl.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-7.667187f +- 0.000001f)
    weights.valueAt(2) should be (-10.912737f +- 0.000001f)
  }

  "Ftrl with L1, L2" should "works fine 3" in {
    val weights = Tensor[Float](T(1.0f, 2.0f))
    val grads = Tensor[Float](T(0.1f, 0.2f))
    val ftrl = new Ftrl[Float](3.0, l1RegularizationStrength = 0.001,
      l2RegularizationStrength = 2.0)
    (1 to 10).foreach{_ =>
      ftrl.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-0.240599f +- 0.000001f)
    weights.valueAt(2) should be (-0.468293f +- 0.000001f)
  }

  "Ftrl with L1, L2, L2Shrinkage" should "works fine 3" in {
    val weights = Tensor[Float](T(1.0f, 2.0f))
    val grads = Tensor[Float](T(0.1f, 0.2f))
    val ftrl = new Ftrl[Float](3.0, initialAccumulatorValue = 0.1, l1RegularizationStrength = 0.001,
      l2RegularizationStrength = 2.0, l2ShrinkageRegularizationStrength = 0.1f)
    (1 to 10).foreach{_ =>
      ftrl.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-0.219319f +- 0.000001f)
    weights.valueAt(2) should be (-0.406429f +- 0.000001f)
  }

  "Ftrl save/load" should "works fine" in {
    val weights = Tensor[Float](T(1.0f, 2.0f))
    val grads = Tensor[Float](T(0.1f, 0.2f))
    val ftrl = new Ftrl[Float](3.0, initialAccumulatorValue = 0.1, l1RegularizationStrength = 0.001,
      l2RegularizationStrength = 2.0, l2ShrinkageRegularizationStrength = 0.1f)
    val tmpFile = java.io.File.createTempFile("ftrl", ".optim")
    ftrl.save(tmpFile.getAbsolutePath, true)
    val loaded = OptimMethod.load[Float](tmpFile.getAbsolutePath)

    (1 to 10).foreach{_ =>
      loaded.optimize(_ => (0.0f, grads), weights)
    }

    weights.valueAt(1) should be (-0.219319f +- 0.000001f)
    weights.valueAt(2) should be (-0.406429f +- 0.000001f)

  }
}

