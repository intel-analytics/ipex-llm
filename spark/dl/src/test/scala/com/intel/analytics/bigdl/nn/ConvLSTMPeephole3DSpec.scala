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

package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._

@com.intel.analytics.bigdl.tags.Parallel
class ConvLSTMPeephole3DSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A ConvLSTMPeepwhole3D" should " work in BatchMode" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = Recurrent[Double]()
    val model = Sequential[Double]()
      .add(rec
        .add(ConvLSTMPeephole3D[Double](
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1, withPeephole = true)))

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand

    for (i <- 1 to 3) {
      val output = model.forward(input)
      model.backward(input, output)
    }
  }

  "A ConvLSTMPeepwhole3D" should " return state" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val model = Recurrent[Double]()
        .add(ConvLSTMPeephole3D[Double](
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1, withPeephole = true))

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand

    val output = model.forward(input)

    val state = model.getState()
    val hidden = state.asInstanceOf[Table].apply(1).asInstanceOf[Tensor[Double]]
    hidden.map(output.select(2, seqLength), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
  }

  "ConvLSTMPeephole3D L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 1
    val kernalW = 3
    val kernalH = 3

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val criterion = new TimeDistributedCriterion[Double](new MSECriterion[Double])

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand
    val labels = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3, 3).rand

    val rec = Recurrent[Double]()
    val model1 = Sequential[Double]()
      .add(rec
        .add(ConvLSTMPeephole3D[Double](
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1,
          withPeephole = true)))

    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(ConvLSTMPeephole3D[Double](
          inputSize,
          hiddenSize,
          kernalW, kernalH,
          1,
          wRegularizer = L2Regularizer(0.1),
          uRegularizer = L2Regularizer(0.1),
          bRegularizer = L2Regularizer(0.1),
          cRegularizer = L2Regularizer(0.1),
          withPeephole = true)))

    val (weights2, grad2) = model2.getParameters()
    weights2.copy(weights1.clone())
    grad2.copy(grad1.clone())

    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss1: Array[Double] = null
    for (i <- 1 to 100) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
      println(s"${i}-th loss = ${loss1(0)}")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${i}-th loss = ${loss2(0)}")
    }

    weights1 should be(weights2)
    loss1 should be(loss2)
  }

  "A ConvLSTMPeepwhole " should "work with feedbackOutput correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 5, 5).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2

    val model2 = Recurrent().add(ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1))
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val input2 = Tensor(Array(batchSize, seqLength, inputSize, 5, 5))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val output2 = model2.forward(input2).toTensor
    val gradInput2 = model2.backward(input2, gradOutput).toTensor
    val gradient2 = model2.getParameters()._2

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
    gradInput.map(gradInput2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
    gradient.map(gradient2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
  }
}
