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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class LSTMSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Recurrent LSTM Language Model Module " should "converge" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
    model.add(Recurrent[Double](hiddenSize, bpttTruncate)
      .add(LSTMCell[Double](inputSize, hiddenSize))
      .add(Tanh()))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0)*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    for (i <- 1 to 10000) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }
}
