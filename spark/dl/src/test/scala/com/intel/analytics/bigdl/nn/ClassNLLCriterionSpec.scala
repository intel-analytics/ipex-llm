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
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class ClassNLLCriterionSpec extends FlatSpec with Matchers {
  "A ClassNLL Criterion with -1 label " should "generate correct output and grad" in {
    val criterion = new ClassNLLCriterion[Double]()
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.0262627674932
    input(Array(1, 2)) = -1.2412600935171
    input(Array(1, 3)) = -1.0423174168648
    input(Array(2, 1)) = -0.90330565804228
    input(Array(2, 2)) = -1.3686840144413
    input(Array(2, 3)) = -1.0778380454479
    input(Array(3, 1)) = -0.99131220658219
    input(Array(3, 2)) = -1.0559142847536
    input(Array(3, 3)) = -1.2692712660404
    val target = Tensor[Double](3)
    target(Array(1)) = -1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 1.31897764024085
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = 0
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -0.5
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -0.5
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }

  "A ClassNLL Criterion " should "generate correct output and grad" in {
    val criterion = new ClassNLLCriterion[Double]()
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.0262627674932
    input(Array(1, 2)) = -1.2412600935171
    input(Array(1, 3)) = -1.0423174168648
    input(Array(2, 1)) = -0.90330565804228
    input(Array(2, 2)) = -1.3686840144413
    input(Array(2, 3)) = -1.0778380454479
    input(Array(3, 1)) = -0.99131220658219
    input(Array(3, 2)) = -1.0559142847536
    input(Array(3, 3)) = -1.2692712660404
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 1.2214060159916
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -0.33333333333333
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -0.33333333333333
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -0.33333333333333
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
  "A ClassNLL Criterion with sizeAverage False" should "generate correct output and grad" in {
    val criterion = new ClassNLLCriterion[Double](null, false)
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.10821131127
    input(Array(1, 2)) = -0.92179085988591
    input(Array(1, 3)) = -1.3017876357682
    input(Array(2, 1)) = -0.72992115377362
    input(Array(2, 2)) = -1.2817109257719
    input(Array(2, 3)) = -1.4250730090114
    input(Array(3, 1)) = -1.1074577039332
    input(Array(3, 2)) = -1.0506933510994
    input(Array(3, 3)) = -1.1397251596433
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 3.5296473966852
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -1
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -1
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -1
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }

  "A ClassNLL Criterion with weight" should "generate correct output and grad" in {
    val weight = Tensor[Double](3)
    weight(Array(1)) = 0.35054216370918
    weight(Array(2)) = 0.76185464672744
    weight(Array(3)) = 0.66953149507754
    val criterion = new ClassNLLCriterion[Double](weight, false)
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.1894985426003
    input(Array(1, 2)) = -1.1789041748521
    input(Array(1, 3)) = -0.94672288864566
    input(Array(2, 1)) = -0.70491562360676
    input(Array(2, 2)) = -1.3937761580642
    input(Array(2, 3)) = -1.3559084361956
    input(Array(3, 1)) = -1.0404241993415
    input(Array(3, 2)) = -1.0287857984981
    input(Array(3, 3)) = -1.240448289816
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 2.309343433418
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -0.35054216370918
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -0.76185464672744
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -0.66953149507754
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }
  "A ClassNLL Criterion with weight and sizeAverage" should "generate correct output and grad" in {
    val weight = Tensor[Double](3)
    weight(Array(1)) = 0.539598016534
    weight(Array(2)) = 0.20644677849486
    weight(Array(3)) = 0.67927200254053
    val criterion = new ClassNLLCriterion[Double](weight)
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = -1.2412808758149
    input(Array(1, 2)) = -1.4300331461186
    input(Array(1, 3)) = -0.75144359487463
    input(Array(2, 1)) = -1.2200775853117
    input(Array(2, 2)) = -1.1747087276299
    input(Array(2, 3)) = -0.92663456371434
    input(Array(3, 1)) = -1.1718541533533
    input(Array(3, 2)) = -1.0983546295516
    input(Array(3, 3)) = -1.0306113735619
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3
    val expectedOutput = 1.1312383221403
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -0.37858111084791
    expectedGrad(Array(1, 2)) = 0
    expectedGrad(Array(1, 3)) = 0
    expectedGrad(Array(2, 1)) = 0
    expectedGrad(Array(2, 2)) = -0.14484273169791
    expectedGrad(Array(2, 3)) = 0
    expectedGrad(Array(3, 1)) = 0
    expectedGrad(Array(3, 2)) = 0
    expectedGrad(Array(3, 3)) = -0.47657615745419
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    assert(abs(expectedOutput - output) < 1e-6)
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
  }

  "A ClassNLL Criterion with 1d input" should "generate correct output and grad" in {
    val weightData: Array[Double] = Array(
      0.86300174100325, 0.39250248204917, 0.73511490179226
    )
    val weight = Tensor[Double](Storage(weightData), 1, Array(3))
    val criterion = new ClassNLLCriterion[Double](weight)
    val inputData: Array[Double] = Array(
      -0.80726062008062, -1.5266720155708, -1.0886697225727
    )
    val input = Tensor[Double](Storage(inputData), 1, Array(3))
    val target = Tensor[Double](1)
    target(Array(1)) = 2
    val expectedOutput = 1.5266720155708
    val expectedGradData: Array[Double] = Array(
      0, -1, 0
    )
    val expectedGrad = Tensor[Double](Storage(expectedGradData), 1, Array(3))
    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)
    output should be(expectedOutput +- 1e-6)
    gradInput.map(expectedGrad, (v1, v2) => {
      v1 should be(v2 +- 1e-6);
      v1
    })
  }

  "A ClassNLL Criterion with probabilities input" should "generate correct output and grad" in {

    val input = Tensor[Float](Array(4, 4)).rand()
    val target = Tensor[Float](Array[Float](1, 2, 3, 4), Array(4))

    val logSoftMax = LogSoftMax[Float]()
    val softMax = SoftMax[Float]()

    val logProb = logSoftMax.forward(input)
    val prob = softMax.forward(input)

    val referenceLayer = ClassNLLCriterion[Float]()
    val testedLayer = ClassNLLCriterion[Float](logProbAsInput = false)

    val expectedLoss = referenceLayer.forward(logProb, target)
    val loss = testedLayer.forward(prob, target)

    val expectedGradInput = logSoftMax.backward(input, referenceLayer.backward(logProb, target))
    val gradInput = softMax.backward(input, testedLayer.backward(prob, target))

    math.abs(expectedLoss - loss) < 1e-5 should be (true)
    expectedGradInput.almostEqual(gradInput, 1e-5) should be (true)
  }

  "A ClassNLL Criterion with probabilities input 1d" should "generate correct output and grad" in {

    val input = Tensor[Float](Array(4)).rand()
    val target = Tensor[Float](Array[Float](4), Array(1))

    val logSoftMax = LogSoftMax[Float]()
    val softMax = SoftMax[Float]()

    val logProb = logSoftMax.forward(input)
    val prob = softMax.forward(input)

    val referenceLayer = ClassNLLCriterion[Float]()
    val testedLayer = ClassNLLCriterion[Float](logProbAsInput = false)

    val expectedLoss = referenceLayer.forward(logProb, target)
    val loss = testedLayer.forward(prob, target)

    val expectedGradInput = logSoftMax.backward(input, referenceLayer.backward(logProb, target))
    val gradInput = softMax.backward(input, testedLayer.backward(prob, target))

    math.abs(expectedLoss - loss) < 1e-5 should be (true)
    expectedGradInput.almostEqual(gradInput, 1e-5) should be (true)
  }
}
