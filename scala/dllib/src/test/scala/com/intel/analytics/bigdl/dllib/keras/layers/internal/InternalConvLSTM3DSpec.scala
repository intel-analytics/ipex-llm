/*
 * Copyright 2018 Analytics Zoo Authors.
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
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.pipeline.api.keras.layers.InternalConvLSTM3D
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalRecurrent
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._

class InternalConvLSTM3DSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A InternalConvLSTM3D " should " work in BatchMode" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernal = 2
    val rec = new InternalRecurrent[Double]()
    val model = Sequential[Double]()
      .add(rec
        .add(new InternalConvLSTM3D[Double](
          inputSize,
          hiddenSize,
          kernal,
          padding = -1,
          withPeephole = true)))

    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5, 5).rand

    for (i <- 1 to 3) {
      val output = model.forward(input).toTensor[Double]
      for((value, j) <- output.size.view.zipWithIndex) {
        if (j > 2) {
          require(value == input.size(j + 1))
        }
      }
      model.backward(input, output)
    }
  }

  "A InternalConvLSTM3D " should "work with valid padding" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 1

    val input = Tensor[Double](batchSize, seqLength, inputSize, 16, 16, 16).rand()

    val rec = new InternalRecurrent[Double]()
    val model = Sequential[Double]()
      .add(rec
        .add(new InternalConvLSTM3D[Double](inputSize, hiddenSize, 4,
          padding = 0, withPeephole = false)))

    val output = model.forward(input)
    model.backward(input, output)
  }

  "A InternalConvLSTM3D" should " return state" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernal = 3
    val model = new InternalRecurrent[Double]()
      .add(new InternalConvLSTM3D[Double](
        inputSize,
        hiddenSize,
        kernal,
        padding = -1,
        withPeephole = true))

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand

    val output = model.forward(input)

    val state = model.getHiddenState()
    val hidden = state.asInstanceOf[Table].apply(1).asInstanceOf[Tensor[Double]]
    hidden.map(output.select(2, seqLength), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
  }

  "InternalConvLSTM3D L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 1
    val kernal = 3

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val criterion = new TimeDistributedCriterion[Double](new MSECriterion[Double])

    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3, 3).rand
    val labels = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3, 3).rand

    val rec = new InternalRecurrent[Double]()
    val model1 = Sequential[Double]()
      .add(rec
        .add(new InternalConvLSTM3D[Double](
          inputSize,
          hiddenSize,
          kernal,
          padding = -1,
          withPeephole = true)))

    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential[Double]()
      .add(new InternalRecurrent[Double]()
        .add(new InternalConvLSTM3D[Double](
          inputSize,
          hiddenSize,
          kernal,
          padding = -1,
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
}
