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

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._

@com.intel.analytics.bigdl.tags.Serial
class RecurrentSpec extends FlatSpec with Matchers {

  "A Recurrent" should " converge when batchSize changes" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val time = 4
    val batchSize1 = 5
    val batchSize2 = 8
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
      .add(Select(2, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    val input1 = Tensor[Double](Array(batchSize1, time, inputSize)).rand
    val input2 = Tensor[Double](batchSize2, time, inputSize).rand

    val gradOutput1 = Tensor[Double](batchSize1, outputSize).rand
    val gradOutput2 = Tensor[Double](batchSize2, outputSize).rand

    model.clearState()

    model.forward(input1)
    model.backward(input1, gradOutput1)
    val gradInput1 =
      Tensor[Double](batchSize1, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output1 = Tensor[Double](batchSize1, outputSize).copy(model.output.toTensor[Double])

    model.clearState()

    model.forward(input2)
    model.backward(input2, gradOutput2)
    val gradInput2 =
      Tensor[Double](batchSize2, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output2 = Tensor[Double](batchSize2, outputSize).copy(model.output.toTensor[Double])

    model.forward(input1)
    model.backward(input1, gradOutput1)
    val gradInput1compare =
      Tensor[Double](batchSize1, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output1compare = Tensor[Double](batchSize1, outputSize).copy(model.output.toTensor[Double])

    model.forward(input2)
    model.backward(input2, gradOutput2)
    val gradInput2compare =
      Tensor[Double](batchSize2, time, inputSize).copy(model.gradInput.toTensor[Double])
    val output2compare = Tensor[Double](batchSize2, outputSize).copy(model.output.toTensor[Double])

    model.hashCode()

    output1 should be (output1compare)
    output2 should be (output2compare)

    gradInput1 should be (gradInput1compare)
    gradInput2 should be (gradInput2compare)
  }

  "A Recurrent Language Model Module " should "converge" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh[Double]())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

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

    for (i <- 1 to 50) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A Recurrent Module " should "converge in batch mode" in {

    val batchSize = 10
    val nWords = 5
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(2, nWords))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

    val input = Tensor[Double](Array(batchSize, nWords, inputSize))
    val labels = Tensor[Double](batchSize)
    for (b <- 1 to batchSize) {
      for (i <- 1 to nWords) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
      }
      val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
      labels.setValue(b, rdmLabel)
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

    for (i <- 1 to 50) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(s"${i}-th loss = ${loss(0)}")
    }

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A Recurrent Module " should "perform correct gradient check" in {

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(RnnCell[Double](inputSize, hiddenSize, Tanh())))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(Math.random()*inputSize).toInt
      val rdmInput = Math.ceil(Math.random()*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkLayer[Double](model, input)
    println("gradient check for weights")
    val gradCheck = new GradientCheckerRNN(1e-2, 1)
    val checkFlag = gradCheck.checkLayer(model, input, labels)
  }

  "Recurrent dropout" should "work correclty" in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 1

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(1, seqLength, inputSize))
    for (i <- 1 to seqLength) {
      val rdmInput = 3
      input.setValue(1, i, rdmInput, 1.0)
    }

    println(input)
    val gru = GRU[Double](inputSize, hiddenSize, 0.2)
    val model = Recurrent[Double]().add(gru)

    val field = model.getClass.getDeclaredField("cells")
    field.setAccessible(true)
    val cells = field.get(model).asInstanceOf[ArrayBuffer[Cell[Double]]]

    val dropoutsRecurrent = model.asInstanceOf[Container[_, _, Double]].findModules("Dropout")
    val dropoutsCell = gru.cell.asInstanceOf[Container[_, _, Double]].findModules("Dropout")
    val dropouts = dropoutsRecurrent ++ dropoutsCell
    dropouts.size should be (6)

    val output = model.forward(input)
    val noises1 = dropouts.map(d => d.asInstanceOf[Dropout[Double]].noise.clone())
    noises1(0) should not be noises1(1)

    val noises = dropoutsCell.map(d => d.asInstanceOf[Dropout[Double]].noise.clone())
    for (i <- dropoutsCell.indices) {
      cells.foreach(c => {
        val noise = c.cell.asInstanceOf[Container[_, _, Double]]
          .findModules("Dropout")(i)
          .asInstanceOf[Dropout[Double]]
          .noise
        noise should be(noises(i))
      })
    }


    model.forward(input)

    var flag = true
    for (i <- dropoutsCell.indices) {
      cells.foreach(c => {
        val newNoises = c.cell.asInstanceOf[Container[_, _, Double]]
          .findModules("Dropout")
        val noise = newNoises(i).asInstanceOf[Dropout[Double]].noise
        flag = flag && (noise == noises(i))
      })
    }

    flag should be (false)
  }

  "A Recurrent Module " should "work with getFinalStateAndCell " in {
    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    val batchSize = 1
    val time = 4
    RNG.setSeed(seed)

    val rec = Recurrent[Double]()
      .add(RnnCell[Double](inputSize, hiddenSize, Tanh()))
    val model = Sequential[Double]()
      .add(rec)

    val input = Tensor[Double](Array(batchSize, time, inputSize)).rand

    val output = model.forward(input).asInstanceOf[Tensor[Double]]
    val (finalState, cellStatus) = rec.getFinalStateAndCellStatus()

    finalState.map(output.asInstanceOf[Tensor[Double]].select(2, time), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })

    assert(cellStatus == null, "rnn should not have cell state")
  }
}
