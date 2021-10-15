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

package com.intel.analytics.bigdl.dllib.keras.objectives

import com.intel.analytics.bigdl.dllib.nn.{LogSoftMax, SoftMax}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.keras.layers.{KerasRunner, Loss}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers.KerasBaseSpec

import scala.math.abs

class SparseCategoricalCrossEntropySpec extends KerasBaseSpec {

  "SparseCategoricalCrossEntropy" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, ])
        |target_tensor = Input(batch_shape=[3, ])
        |loss = sparse_categorical_crossentropy(target_tensor, input_tensor)
        |input = input = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        |Y = np.array([0.0, 1.0, 2.0])
      """.stripMargin
    val loss = SparseCategoricalCrossEntropy[Float](logProbAsInput = false)
    val (gradInput, gradWeight, weights, input, target, output) =
      KerasRunner.run(kerasCode, Loss)
    val boutput = loss.forward(input, target)
    val koutput = output.mean()
    NumericFloat.nearlyEqual(boutput, koutput, 1e-5) should be (true)
  }

  "SparseCategoricalCrossEntropy" should "generate correct output and grad" in {
    val criterion = SparseCategoricalCrossEntropy[Double](logProbAsInput = true)
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
    target(Array(1)) = 0
    target(Array(2)) = 1
    target(Array(3)) = 2
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
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "SparseCategoricalCrossEntropy with weight" should "generate correct output and grad" in {
    val weight = Tensor[Double](3)
    weight(Array(1)) = 0.539598016534
    weight(Array(2)) = 0.20644677849486
    weight(Array(3)) = 0.67927200254053
    val criterion = SparseCategoricalCrossEntropy[Double](
      weights = weight, logProbAsInput = true)
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
    target(Array(1)) = 0
    target(Array(2)) = 1
    target(Array(3)) = 2
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
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "SparseCategoricalCrossEntropy with sizeAverage false and 1-based label" should
    "generate correct output and grad" in {
    val criterion = SparseCategoricalCrossEntropy[Double](
      zeroBasedLabel = false, sizeAverage = false, logProbAsInput = true)
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
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "SparseCategoricalCrossEntropy with probabilities input" should
    "generate correct output and grad" in {
    val input = Tensor[Float](Array(4, 4)).rand()
    val target = Tensor[Float](Array[Float](0, 1, 2, 3), Array(4))

    val logSoftMax = LogSoftMax[Float]()
    val softMax = SoftMax[Float]()

    val logProb = logSoftMax.forward(input)
    val prob = softMax.forward(input)

    val referenceLayer = SparseCategoricalCrossEntropy[Float](logProbAsInput = true)
    val testedLayer = SparseCategoricalCrossEntropy[Float]()

    val expectedLoss = referenceLayer.forward(logProb, target)
    val loss = testedLayer.forward(prob, target)

    val expectedGradInput = logSoftMax.backward(input, referenceLayer.backward(logProb, target))
    val gradInput = softMax.backward(input, testedLayer.backward(prob, target))

    math.abs(expectedLoss - loss) < 1e-5 should be (true)
    expectedGradInput.almostEqual(gradInput, 1e-5) should be (true)
  }

}
