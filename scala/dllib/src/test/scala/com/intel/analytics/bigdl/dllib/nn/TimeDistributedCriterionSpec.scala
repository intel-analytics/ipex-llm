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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.math._

class TimeDistributedCriterionSpec extends FlatSpec with Matchers {
  "A ClassNLL Criterion with sizeAverage True and TimeDistributedCriterion sizeAverage True" should
    "generate correct output and grad" in {
    val criterion = ClassNLLCriterion[Double]()
    val layer = TimeDistributedCriterion[Double](criterion, true)

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    val expectedOutput = 1.2214060159916
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -0.16666666666666666
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -0.16666666666666666
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -0.16666666666666666
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -0.16666666666666666
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = -0.16666666666666666
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -0.16666666666666666
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A ClassNLL Criterion with sizeAverage True and TimeDistributedCriterion sizeAverage False" should
    "generate correct output and grad" in {
    val criterion = ClassNLLCriterion[Double]()
    val layer = TimeDistributedCriterion[Double](criterion)

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    val expectedOutput = 2.4428120319832
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -0.3333333333333333
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -0.3333333333333333
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -0.3333333333333333
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -0.3333333333333333
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = -0.3333333333333333
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -0.3333333333333333
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A ClassNLL Criterion with sizeAverage False and TimeDistributedCriterion sizeAverage True" should
    "generate correct output and grad" in {
    val criterion = ClassNLLCriterion[Double](null, false)
    val layer = TimeDistributedCriterion[Double](criterion, true)

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    val expectedOutput = 3.6642180479748996
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -0.5
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -0.5
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -0.5
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -0.5
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = -0.5
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -0.5
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A ClassNLL Criterion with sizeAverage False, TimeDistributedCriterion sizeAverage False" should
    "generate correct output and grad" in {
    val criterion = ClassNLLCriterion[Double](null, false)
    val layer = TimeDistributedCriterion[Double](criterion)

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    val expectedOutput = 7.3284360959496
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -1
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -1
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -1
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -1
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = -1
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -1
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A BCE Criterion" should "generate correct output and grad" in {
    val criterion = BCECriterion[Double]()
    val layer = TimeDistributedCriterion[Double](criterion, true)

    val output = Tensor[Double](3, 2)
    output(Array(1, 1)) = 0.4
    output(Array(1, 2)) = 0.4
    output(Array(2, 1)) = 0.5
    output(Array(2, 2)) = 0.5
    output(Array(3, 1)) = 0.6
    output(Array(3, 2)) = 0.6
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 0
    target(Array(1, 2)) = 0
    target(Array(2, 1)) = 1
    target(Array(2, 2)) = 1
    target(Array(3, 1)) = 1
    target(Array(3, 2)) = 1

    val loss = layer.forward(output, target)
    loss should be(0.57159947 +- 1e-8)

    val gradInput = layer.backward(output, target)
    gradInput(Array(1, 1)) should be(0.2778 +- 0.0001)
    gradInput(Array(1, 2)) should be(0.2778 +- 0.0001)
    gradInput(Array(2, 1)) should be(-0.3333 +- 0.0001)
    gradInput(Array(2, 2)) should be(-0.3333 +- 0.0001)
    gradInput(Array(3, 1)) should be(-0.2778 +- 0.0001)
    gradInput(Array(3, 2)) should be(-0.2778 +- 0.0001)
  }

  "A CrossEntropy Criterion" should "generate correct output and grad" in {
    val criterion = CrossEntropyCriterion[Double]()
    val layer = TimeDistributedCriterion[Double](criterion, true)

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = 0.33655226649716
    input(Array(1, 1, 2)) = 0.77367000770755
    input(Array(1, 1, 3)) = 0.031494265655056
    input(Array(1, 2, 1)) = 0.33655226649716
    input(Array(1, 2, 2)) = 0.77367000770755
    input(Array(1, 2, 3)) = 0.031494265655056
    input(Array(2, 1, 1)) = 0.11129087698646
    input(Array(2, 1, 2)) = 0.14688249188475
    input(Array(2, 1, 3)) = 0.49454387230799
    input(Array(2, 2, 1)) = 0.11129087698646
    input(Array(2, 2, 2)) = 0.14688249188475
    input(Array(2, 2, 3)) = 0.49454387230799
    input(Array(3, 1, 1)) = 0.45682632108219
    input(Array(3, 1, 2)) = 0.85653987620026
    input(Array(3, 1, 3)) = 0.42569971177727
    input(Array(3, 2, 1)) = 0.45682632108219
    input(Array(3, 2, 2)) = 0.85653987620026
    input(Array(3, 2, 3)) = 0.42569971177727

    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(2, 1)) = 2
    target(Array(3, 1)) = 3
    target(Array(1, 2)) = 1
    target(Array(2, 2)) = 2
    target(Array(3, 2)) = 3

    val expectedOutput = 1.2267281042702334

    val loss = layer.forward(input, target)
    loss should be(expectedOutput +- 1e-8)

    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -0.23187185 / 2
    expectedGrad(Array(1, 1, 2)) = 0.15708656 / 2
    expectedGrad(Array(1, 1, 3)) = 0.07478529 / 2
    expectedGrad(Array(2, 1, 1)) = 0.09514888 / 2
    expectedGrad(Array(2, 1, 2)) = -0.23473696 / 2
    expectedGrad(Array(2, 1, 3)) = 0.13958808 / 2
    expectedGrad(Array(3, 1, 1)) = 0.09631823 / 2
    expectedGrad(Array(3, 1, 2)) = 0.14364876 / 2
    expectedGrad(Array(3, 1, 3)) = -0.23996699 / 2
    expectedGrad(Array(1, 2, 1)) = -0.23187185 / 2
    expectedGrad(Array(1, 2, 2)) = 0.15708656 / 2
    expectedGrad(Array(1, 2, 3)) = 0.07478529 / 2
    expectedGrad(Array(2, 2, 1)) = 0.09514888 / 2
    expectedGrad(Array(2, 2, 2)) = -0.23473696 / 2
    expectedGrad(Array(2, 2, 3)) = 0.13958808 / 2
    expectedGrad(Array(3, 2, 1)) = 0.09631823 / 2
    expectedGrad(Array(3, 2, 2)) = 0.14364876 / 2
    expectedGrad(Array(3, 2, 3)) = -0.23996699 / 2
    val gradInput = layer.backward(input, target)

    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MSE Criterion with sizeAverage True and TimeDistributedCriterion sizeAverage True" should
    "generate correct output and grad" in {
    val criterion = MSECriterion[Double]()
    val layer = TimeDistributedCriterion[Double](criterion, true)

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

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MSE Criterion with sizeAverage False and TimeDistributedCriterion sizeAverage True" should
    "generate correct output and grad" in {
    val criterion = MSECriterion[Double]()
    criterion.sizeAverage = false
    val layer = TimeDistributedCriterion[Double](criterion, true)

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

    val expectedOutput = 0.5809751749326332
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = -0.50721812341362 / 2
    expectedGrad(Array(1, 1, 2)) = -1.2595179299824 / 2
    expectedGrad(Array(1, 2, 1)) = 1.1238703178242 / 2
    expectedGrad(Array(1, 2, 2)) = 0.32083974452689 / 2
    expectedGrad(Array(2, 1, 1)) = -0.56642062868923 / 2
    expectedGrad(Array(2, 1, 2)) = 0.8510562335141 / 2
    expectedGrad(Array(2, 2, 1)) = -0.20184816932306 / 2
    expectedGrad(Array(2, 2, 2)) = 0.59350770944729 / 2

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MSE Criterion with sizeAverage False and TimeDistributedCriterion sizeAverage False" should
    "generate correct output and grad" in {
    val criterion = MSECriterion[Double]()
    criterion.sizeAverage = false
    val layer = TimeDistributedCriterion[Double](criterion, false)

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

    val output = layer.forward(input, target)
    val gradInput = layer.backward(input, target)

    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MultiLabelSoftMargin Criterion" should "generate correct output and grad" in {
    val criterion = MultiLabelSoftMarginCriterion[Double]()
    val layer = TimeDistributedCriterion[Double](criterion, true)

    val output = Tensor[Double](3, 2)
    output(Array(1, 1)) = 0.4
    output(Array(2, 1)) = 0.5
    output(Array(3, 1)) = 0.6
    output(Array(1, 2)) = 0.4
    output(Array(2, 2)) = 0.5
    output(Array(3, 2)) = 0.6
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 0
    target(Array(2, 1)) = 1
    target(Array(3, 1)) = 1
    target(Array(1, 2)) = 0
    target(Array(2, 2)) = 1
    target(Array(3, 2)) = 1

    val loss = layer.forward(output, target)
    loss should be(0.608193395686766 +- 1e-8)

    val gradInput = layer.backward(output, target)
    gradInput(Array(1, 1)) should be(0.19956255336948944 / 2 +- 0.0001)
    gradInput(Array(2, 1)) should be(-0.12584688959851295 / 2 +- 0.0001)
    gradInput(Array(3, 1)) should be(-0.11811456459055192 / 2 +- 0.0001)
    gradInput(Array(1, 1)) should be(0.19956255336948944 / 2 +- 0.0001)
    gradInput(Array(2, 1)) should be(-0.12584688959851295 / 2 +- 0.0001)
    gradInput(Array(3, 1)) should be(-0.11811456459055192 / 2 +- 0.0001)
  }

  "A Parallel Criterion" should "generate correct output" in {
    val criterion = new ParallelCriterion[Double]()

    val input = Tensor[Double](3, 2, 3)
    input(Array(1, 1, 1)) = -1.0262627674932
    input(Array(1, 1, 2)) = -1.2412600935171
    input(Array(1, 1, 3)) = -1.0423174168648
    input(Array(1, 2, 1)) = -1.0262627674932
    input(Array(1, 2, 2)) = -1.2412600935171
    input(Array(1, 2, 3)) = -1.0423174168648
    input(Array(2, 1, 1)) = -0.90330565804228
    input(Array(2, 1, 2)) = -1.3686840144413
    input(Array(2, 1, 3)) = -1.0778380454479
    input(Array(2, 2, 1)) = -0.90330565804228
    input(Array(2, 2, 2)) = -1.3686840144413
    input(Array(2, 2, 3)) = -1.0778380454479
    input(Array(3, 1, 1)) = -0.99131220658219
    input(Array(3, 1, 2)) = -1.0559142847536
    input(Array(3, 1, 3)) = -1.2692712660404
    input(Array(3, 2, 1)) = -0.99131220658219
    input(Array(3, 2, 2)) = -1.0559142847536
    input(Array(3, 2, 3)) = -1.2692712660404
    val target = Tensor[Double](3, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 2
    target(Array(2, 2)) = 2
    target(Array(3, 1)) = 3
    target(Array(3, 2)) = 3

    val expectedOutput = 1.2214060159916
    val expectedGrad = Tensor[Double](3, 2, 3)
    expectedGrad(Array(1, 1, 1)) = -0.16666666666666666
    expectedGrad(Array(1, 1, 2)) = 0
    expectedGrad(Array(1, 1, 3)) = 0
    expectedGrad(Array(1, 2, 1)) = -0.16666666666666666
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(1, 2, 3)) = 0
    expectedGrad(Array(2, 1, 1)) = 0
    expectedGrad(Array(2, 1, 2)) = -0.16666666666666666
    expectedGrad(Array(2, 1, 3)) = 0
    expectedGrad(Array(2, 2, 1)) = 0
    expectedGrad(Array(2, 2, 2)) = -0.16666666666666666
    expectedGrad(Array(2, 2, 3)) = 0
    expectedGrad(Array(3, 1, 1)) = 0
    expectedGrad(Array(3, 1, 2)) = 0
    expectedGrad(Array(3, 1, 3)) = -0.16666666666666666
    expectedGrad(Array(3, 2, 1)) = 0
    expectedGrad(Array(3, 2, 2)) = 0
    expectedGrad(Array(3, 2, 3)) = -0.16666666666666666

    val nll = ClassNLLCriterion[Double]()
    val layer1 = TimeDistributedCriterion[Double](nll, true)
    criterion.add(layer1, 1)

    val output = criterion.forward(T(input), T(target))
    val gradInput = criterion.backward(T(input), T(target))

    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput(1), (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }
}
