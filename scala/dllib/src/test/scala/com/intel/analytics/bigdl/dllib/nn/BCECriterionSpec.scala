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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class BCECriterionSpec extends FlatSpec with Matchers {

  "BCECriterion " should "return return right output and gradInput" in {
    val criterion = new BCECriterion[Double]()
    val output = Tensor[Double](3)
    output(Array(1)) = 0.4
    output(Array(2)) = 0.5
    output(Array(3)) = 0.6
    val target = Tensor[Double](3)
    target(Array(1)) = 0
    target(Array(2)) = 1
    target(Array(3)) = 1

    val loss = criterion.forward(output, target)
    loss should be(0.57159947 +- 1e-8)
    val gradInput = criterion.backward(output, target)
    gradInput(Array(1)) should be(0.5556 +- 0.0001)
    gradInput(Array(2)) should be(-0.6667 +- 0.0001)
    gradInput(Array(3)) should be(-0.5556 +- 0.0001)

  }

  "BCECriterion's eps" should "works" in {
    val criterion = BCECriterion[Float]()
    val output = Tensor[Float](3)
    output.setValue(1, 0f)
    output.setValue(2, 1f)
    output.setValue(3, 0.5f)
    val target = Tensor[Float](3)
    target.setValue(1, 0)
    target.setValue(2, 1)
    target.setValue(3, 1)

    val loss = criterion.forward(output, target)
    java.lang.Float.isNaN(loss) should be (false)
  }

  "BCECriterion's eps with weight" should "works" in {
    val weights = Tensor[Float](3).rand()
    val criterion = BCECriterion[Float](weights)
    val output = Tensor[Float](3)
    output.setValue(1, 0f)
    output.setValue(2, 1f)
    output.setValue(3, 0.5f)
    val target = Tensor[Float](3)
    target.setValue(1, 0)
    target.setValue(2, 1)
    target.setValue(3, 1)

    val loss = criterion.forward(output, target)
    java.lang.Float.isNaN(loss) should be (false)
  }

  "BCECriterion with more than two dimensions small input" should "" +
    "return return right output and gradInput" in {

    val weights = Tensor[Double](3, 2, 2).rand()
    val criterion = new BCECriterion[Double](weights)
    val input = Tensor[Double](4, 3, 2, 2).rand()
    val target = Tensor[Double](4, 3, 2, 2).rand()

    val weightsRef = Tensor[Double]().resizeAs(weights).copy(weights).reshape(Array(3 * 2 * 2))
    val criterionRef = new BCECriterion[Double](weightsRef)
    val inputRef = Tensor[Double]().resizeAs(input).copy(input).reshape(Array(4, 3 * 2 * 2))
    val targetRef = Tensor[Double]().resizeAs(target).copy(target).reshape(Array(4, 3 * 2 * 2))

    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target).clone()

    val outputRef = criterionRef.forward(inputRef, targetRef)
    val gradInputRef = criterionRef.backward(inputRef, targetRef).clone()

    output should be (outputRef +- 1e-7)
    gradInput.almostEqual(gradInputRef, 1e-7) should be (true)

  }

  "BCECriterion with more than two dimensions large input" should "" +
    "return return right output and gradInput" in {

    val weights = Tensor[Double](3, 32, 32).rand()
    val criterion = new BCECriterion[Double](weights)
    val input = Tensor[Double](4, 3, 32, 32).rand()
    val target = Tensor[Double](4, 3, 32, 32).rand()

    val weightsRef = Tensor[Double]().resizeAs(weights).copy(weights).reshape(Array(3 * 32 * 32))
    val criterionRef = new BCECriterion[Double](weightsRef)
    val inputRef = Tensor[Double]().resizeAs(input).copy(input).reshape(Array(4, 3 * 32 * 32))
    val targetRef = Tensor[Double]().resizeAs(target).copy(target).reshape(Array(4, 3 * 32 * 32))

    val output = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target).clone()

    val outputRef = criterionRef.forward(inputRef, targetRef)
    val gradInputRef = criterionRef.backward(inputRef, targetRef).clone()

    output should be (outputRef +- 1e-7)
    gradInput.almostEqual(gradInputRef, 1e-7) should be (true)

  }

  "Binary LR " should "converge correctly" in {
    def specifiedModel(): Module[Double] = {
      val model = new Sequential[Double]()
      val linear = new Linear[Double](2, 1)
      linear.weight(Array(1, 1)) = 0.1
      linear.weight(Array(1, 2)) = -0.6
      linear.bias(Array(1)) = 0.05
      model.add(linear)
      model.add(new Sigmoid())
      model
    }

    def getTrainModel(): Module[Double] = {
      val model = new Sequential[Double]()
      model.add(new Linear[Double](2, 1))
      model.add(new Sigmoid[Double]())
      model
    }

    def feval(grad: Tensor[Double],
      module: Module[Double],
      criterion: Criterion[Double],
      input: Tensor[Double], target: Tensor[Double])(weights: Tensor[Double])
    : (Double, Tensor[Double]) = {
      module.training()
      grad.zero()
      val output = module.forward(input)
      val loss = criterion.forward(output, target)
      val gradOut = criterion.backward(output, target)
      module.backward(input, gradOut)
      (loss, grad)
    }

    val actualModel = specifiedModel()
    val trainSize = 100000
    val testSize = 10000

    val inputs = Tensor[Double](trainSize, 2)
    val r = new scala.util.Random(1)
    inputs.apply1(v => r.nextDouble())

    val targets = actualModel
      .forward(inputs)
      .toTensor[Double]
      .resize(Array(trainSize))
      .apply1(v => Math.round(v))

    val trainModel = getTrainModel()
    val criterion = new BCECriterion[Double]()
    val (masterWeights, masterGrad) = trainModel.getParameters()
    val optm = new SGD[Double]()
    val config = T("learningRate" -> 10.0, "weightDecay" -> 0.0,
      "momentum" -> 0.0, "learningRateDecay" -> 0.0)

    val batchSize = 1000
    var epoch = 1
    while (epoch < 1000) {
      var i = 1
      var l = 0.0
      while (i <= inputs.size(1)) {
        val (grad, loss) = optm.optimize(feval(masterGrad, trainModel, criterion,
          inputs.narrow(1, i, batchSize),
          targets
            .toTensor[Double]
            .narrow(1, i, batchSize).addSingletonDimension(dim = 2)),
          masterWeights, config, config)
        l += loss(0)
        i += batchSize
      }
      if (l / inputs.size(1) * batchSize < 8e-3) epoch += 1
    }

    val testData = Tensor[Double](testSize, 2)
    testData.apply1(v => r.nextDouble())
    val testTarget = actualModel
      .forward(testData).toTensor[Double].apply1(v => Math.round(v))

    val testResult = trainModel.forward(testData)
      .toTensor[Double].apply1(v => Math.round(v))

    var corrects = 0
    var i = 1
    while (i <= testSize) {
      if (testTarget(Array(i, 1)) == testResult(Array(i, 1))) corrects += 1
      i += 1
    }

    corrects should be(10000)
  }
}
