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

import org.scalatest.FlatSpec
import com.intel.analytics.bigdl.tensor.Tensor

import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class MSECriterionSpec extends FlatSpec {
  "A MSE Criterion" should " be fast" in {
    val mse = MSECriterion[Double]
    val input = Tensor[Double](3, 50, 50)
    val target = Tensor[Double](3, 50, 50)

    val warmUp = 50
    for (i <- 1 to warmUp) {
      mse.forward(input, target)
      mse.backward(input, target)
    }

    val iteration = 50
    var sum = 0.0
    for (i <- 1 to warmUp) {
      val st = System.nanoTime
      mse.forward(input, target)
      mse.backward(input, target)
      val eta = (System.nanoTime() - st) / 1e9
      sum += eta
      println(s"eta = ${eta}")
    }

    println(s"average eta = ${sum / iteration}")
  }
  "A MSE Criterion " should "generate correct output and grad" in {
    val mse = new MSECriterion[Double]
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 0.17503996845335
    input(Array(1, 1, 2)) = 0.83220188552514
    input(Array(1, 2, 1)) = 0.48450597329065
    input(Array(1, 2, 2)) = 0.64701424003579
    input(Array(2, 1, 1)) = 0.62694586534053
    input(Array(2, 1, 2)) = 0.34398410236463
    input(Array(2, 2, 1)) = 0.55356747563928
    input(Array(2, 2, 2)) = 0.20383032318205
    val target = Tensor[Double](2, 2, 2)
    target(Array(1, 1, 1)) = 0.69956525065936
    target(Array(1, 1, 2)) = 0.86074831243604
    target(Array(1, 2, 1)) = 0.54923197557218
    target(Array(1, 2, 2)) = 0.57388074393384
    target(Array(2, 1, 1)) = 0.63334444304928
    target(Array(2, 1, 2)) = 0.99680578662083
    target(Array(2, 2, 1)) = 0.49997645849362
    target(Array(2, 2, 2)) = 0.23869121982716
    val expectedOutput = 0.08947300078144
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = -0.1311313205515
    expectedGrad(Array(1, 1, 2)) = -0.0071366067277268
    expectedGrad(Array(1, 2, 1)) = -0.016181500570383
    expectedGrad(Array(1, 2, 2)) = 0.018283374025486
    expectedGrad(Array(2, 1, 1)) = -0.0015996444271877
    expectedGrad(Array(2, 1, 2)) = -0.16320542106405
    expectedGrad(Array(2, 2, 1)) = 0.013397754286416
    expectedGrad(Array(2, 2, 2)) = -0.0087152241612785
    val output = mse.forward(input, target)
    val gradInput = mse.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }

  "A MSE Criterion with sizeAverage:false " should "generate correct output and grad" in {
    val mse = new MSECriterion[Double]
    mse.sizeAverage = false
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 0.64631252549589
    input(Array(1, 1, 2)) = 0.1541522629559
    input(Array(1, 2, 1)) = 0.6778122568503
    input(Array(1, 2, 2)) = 0.55571207939647
    input(Array(2, 1, 1)) = 0.53701480175368
    input(Array(2, 1, 2)) = 0.83826910308562
    input(Array(2, 2, 1)) = 0.27449130127206
    input(Array(2, 2, 2)) = 0.63781907199882
    val target = Tensor[Double](2, 2, 2)
    target(Array(1, 1, 1)) = 0.8999215872027
    target(Array(1, 1, 2)) = 0.7839112279471
    target(Array(1, 2, 1)) = 0.11587709793821
    target(Array(1, 2, 2)) = 0.39529220713302
    target(Array(2, 1, 1)) = 0.8202251160983
    target(Array(2, 1, 2)) = 0.41274098632857
    target(Array(2, 2, 1)) = 0.37541538593359
    target(Array(2, 2, 2)) = 0.34106521727517
    val expectedOutput = 1.1619503498653
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = -0.50721812341362
    expectedGrad(Array(1, 1, 2)) = -1.2595179299824
    expectedGrad(Array(1, 2, 1)) = 1.1238703178242
    expectedGrad(Array(1, 2, 2)) = 0.32083974452689
    expectedGrad(Array(2, 1, 1)) = -0.56642062868923
    expectedGrad(Array(2, 1, 2)) = 0.8510562335141
    expectedGrad(Array(2, 2, 1)) = -0.20184816932306
    expectedGrad(Array(2, 2, 2)) = 0.59350770944729
    val output = mse.forward(input, target)
    val gradInput = mse.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
}
