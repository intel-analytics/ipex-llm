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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class RecurrentDecoderSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A ConvLSTMPeepwhole forward" should "work with RecurrentDecoder" in {
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

    val model2 = Recurrent().add(ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1))
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val input2 = Tensor(Array(batchSize, seqLength, inputSize, 5, 5))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val output2 = model2.forward(input2).toTensor

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
  }

  "A LSTM " should "work with feedbackOutput correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(LSTM(inputSize, hiddenSize))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    model.backward(input, gradOutput)

    val model2 = Recurrent().add(LSTM(inputSize, hiddenSize))
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val input2 = Tensor(Array(batchSize, seqLength, inputSize))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val output2 = model2.forward(input2).toTensor

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
  }

  "A LSTM " should "count time correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(LSTM(inputSize, hiddenSize))

    var ft = 0L
    var bt = 0L
    (0 until 10).foreach { _ =>
      var st = System.nanoTime()
      model.forward(input)
      ft += System.nanoTime() - st
      st = System.nanoTime()
      model.backward(input, gradOutput)
      bt += System.nanoTime() - st
    }

    val times = model.getTimes()
    val modelFt = times.map(v => v._2).sum
    val modelBt = times.map(v => v._3).sum
    modelFt should be (ft +- ft / 100)
    modelBt should be (bt +- bt / 100)
  }

  "A LSTM " should "reset time correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(LSTM(inputSize, hiddenSize))

    (0 until 10).foreach { _ =>
      model.forward(input)
      model.backward(input, gradOutput)
    }
    model.resetTimes()
    val a = model.getTimes()

    var ft = 0L
    var bt = 0L
    (0 until 10).foreach { _ =>
      var st = System.nanoTime()
      model.forward(input)
      ft += System.nanoTime() - st
      st = System.nanoTime()
      model.backward(input, gradOutput)
      bt += System.nanoTime() - st
    }

    val times = model.getTimes()
    val modelFt = times.map(v => v._2).sum
    val modelBt = times.map(v => v._3).sum
    modelFt should be (ft +- ft / 100)
    modelBt should be (bt +- bt / 100)
  }

  "A LSTMPeepwhole " should "work with feedbackOutput correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 3
    val inputSize = 3
    val seqLength = 2
    val seed = 100
    val batchSize = 1

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand

    val model2 = Recurrent().add(LSTMPeephole(inputSize, hiddenSize))
    model2.getParameters()._1.fill(0.5)

    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(LSTMPeephole(inputSize, hiddenSize))
    model.getParameters()._1.fill(0.5)

    val output = model.forward(input).toTensor

    val input2 = Tensor(Array(batchSize, seqLength, hiddenSize))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val output2 = model2.forward(input2).toTensor
    val gradInput2 = model2.backward(input2, gradOutput).toTensor

    output.map(output2, (v1, v2) => {
      assert(v1 - v2 < 1e-8)
      v1
    })
  }

  "A ConvLSTMPeepwhole backward" should "work with RecurrentDecoder" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 3
    val inputSize = 3
    val seqLength = 2
    val seed = 100
    val batchSize = 2

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize, 3, 3).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2

    val input2 = input.clone()
    input2.resize(batchSize, 1, inputSize, 3, 3)
    val model2 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val model3 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    var i = 0
    while (i < model3.parameters()._1.length) {
      model3.parameters()._1(i).set(model2.parameters()._1(i))
      i += 1
    }
    i = 0
    while (i < model3.parameters()._2.length) {
      model3.parameters()._2(i).set(model2.parameters()._2(i))
      i += 1
    }

    val state = T(Tensor[Double](batchSize, hiddenSize, 3, 3),
      Tensor[Double](batchSize, hiddenSize, 3, 3))
    val output2 = model2.forward(T(input, state))
    val output3 = model3.forward(output2)

    val gradOutput3 = gradOutput.select(2, 2)
    val input3 = output2.clone()
    val tmp = T(input3.toTable[Tensor[Double]](1).squeeze(2), input3.toTable(2))
    val gradInput3 = model3.backward(tmp, T(gradOutput3, state))
    val tmp_gradInput = gradInput3.clone
    tmp_gradInput(1) = gradOutput.select(2, 1).add(gradInput3.toTable[Tensor[Double]](1))
    val gradInput2 = model2.backward(T(input, state), tmp_gradInput)
    val finalOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3)
    finalOutput.narrow(2, 1, 1).copy(output2.toTable[Tensor[Double]](1))
    finalOutput.narrow(2, 2, 1).copy(output3.toTable[Tensor[Double]](1))
    output.map(finalOutput, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })

    gradient.map(model2.getParameters()._2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })

    val newGradInput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3)
    newGradInput.narrow(2, 1, 1).copy(gradInput2.toTable[Tensor[Double]](1))
    newGradInput.narrow(2, 2, 1).copy(gradInput3.toTable[Tensor[Double]](1))
    gradInput.map(newGradInput, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
  }

  "A LSTM backward" should "work with RecurrentDecoder" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 3
    val inputSize = 3
    val seqLength = 2
    val seed = 100
    val batchSize = 2

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(LSTM(inputSize, hiddenSize))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2

    val input2 = input.clone()
    input2.resize(batchSize, 1, inputSize)
    val model2 = LSTM(inputSize, hiddenSize)
    model2.includePreTopology = true
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val model3 = LSTM(inputSize, hiddenSize)
    model3.includePreTopology = true
    var i = 0
    while (i < model3.parameters()._1.length) {
      model3.parameters()._1(i).set(model2.parameters()._1(i))
      i += 1
    }
    i = 0
    while (i < model3.parameters()._2.length) {
      model3.parameters()._2(i).set(model2.parameters()._2(i))
      i += 1
    }

    val state = T(Tensor[Double](batchSize, hiddenSize),
      Tensor[Double](batchSize, hiddenSize))
    val output2 = model2.forward(T(input, state))
    val output3 = model3.forward(output2)

    val gradOutput3 = gradOutput.select(2, 2)
    val input3 = output2.clone()
    val tmp = T(input3.toTable[Tensor[Double]](1).squeeze(2), input3.toTable(2))
    val gradInput3 = model3.backward(tmp, T(gradOutput3, state))
    val tmp_gradInput = gradInput3.clone
    tmp_gradInput(1) = gradOutput.select(2, 1).add(gradInput3.toTable[Tensor[Double]](1))
    val gradInput2 = model2.backward(T(input, state), tmp_gradInput)
    val finalOutput = Tensor[Double](batchSize, seqLength, hiddenSize)
    finalOutput.narrow(2, 1, 1).copy(output2.toTable[Tensor[Double]](1))
    finalOutput.narrow(2, 2, 1).copy(output3.toTable[Tensor[Double]](1))
    output.map(finalOutput, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })

    gradient.map(model2.getParameters()._2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })

    val newGradInput = Tensor[Double](batchSize, seqLength, hiddenSize)
    newGradInput.narrow(2, 1, 1).copy(gradInput2.toTable[Tensor[Double]](1))
    newGradInput.narrow(2, 2, 1).copy(gradInput3.toTable[Tensor[Double]](1))
    gradInput.map(newGradInput, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
  }

  "A LSTM backward with RecurrentDecoder" should "get the same result with updateGradInput" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(LSTM(inputSize, hiddenSize))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2

    val rec2 = RecurrentDecoder(seqLength)
    val model2 = rec2
      .add(LSTM(inputSize, hiddenSize))
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()
    val output2 = model2.forward(input).toTensor
    val gradInput2 = model2.updateGradInput(input, gradOutput).toTensor
    model2.accGradParameters(input, gradOutput)
    val gradient2 = model2.getParameters()._2
    require(gradInput.almostEqual(gradInput2, 1e-8) == true)
    require(gradient.almostEqual(gradient2, 1e-8) == true)
  }

  "A ConvLSTMPeepwhole " should "work with RecurrentDecoder get/setHiddenStates" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 3
    val inputSize = 3
    val seqLength = 2
    val seed = 100
    val batchSize = 2

    val initStates = T(Tensor(batchSize, hiddenSize, 3, 3).rand(),
      Tensor(batchSize, hiddenSize, 3, 3).rand())

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize, 3, 3).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3).rand
    val rec = RecurrentDecoder(seqLength)
    val model = rec
      .add(ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1))

    rec.setHiddenState(initStates)
    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2
    val statesGet = rec.getHiddenState().toTable

    val input2 = input.clone()
    input2.resize(batchSize, 1, inputSize, 3, 3)
    val model2 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val model3 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    var i = 0
    while (i < model3.parameters()._1.length) {
      model3.parameters()._1(i).set(model2.parameters()._1(i))
      i += 1
    }
    i = 0
    while (i < model3.parameters()._2.length) {
      model3.parameters()._2(i).set(model2.parameters()._2(i))
      i += 1
    }

    val state = initStates
    val output2 = model2.forward(T(input, state))
    val output3 = model3.forward(output2)

    val gradState = T(Tensor(batchSize, hiddenSize, 3, 3), Tensor(batchSize, hiddenSize, 3, 3))
    val gradOutput3 = gradOutput.select(2, 2)
    val input3 = output2.clone()
    val tmp = T(input3.toTable[Tensor[Double]](1).squeeze(2), input3.toTable(2))
    val gradInput3 = model3.backward(tmp, T(gradOutput3, gradState))
    val tmp_gradInput = gradInput3.clone
    tmp_gradInput(1) = gradOutput.select(2, 1).add(gradInput3.toTable[Tensor[Double]](1))
    val gradInput2 = model2.backward(T(input, state), tmp_gradInput)
    val finalOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3)
    finalOutput.narrow(2, 1, 1).copy(output2.toTable[Tensor[Double]](1))
    finalOutput.narrow(2, 2, 1).copy(output3.toTable[Tensor[Double]](1))
    output.map(finalOutput, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })

    val states1 = statesGet.getState()
    val states2 = output3.toTable[Table](2)
    for (k <- states1.keys) {
      val t1 = states1(k).asInstanceOf[Tensor[Double]]
      val t2 = states2(k).asInstanceOf[Tensor[Double]]
      t1.map(t2, (v1, v2) => {
        assert(abs(v1 - v2) <= 1e-8)
        v1
      })
    }

    gradient.map(model2.getParameters()._2, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })

    val newGradInput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3)
    newGradInput.narrow(2, 1, 1).copy(gradInput2.toTable[Tensor[Double]](1))
    newGradInput.narrow(2, 2, 1).copy(gradInput3.toTable[Tensor[Double]](1))
    gradInput.map(newGradInput, (v1, v2) => {
      assert(abs(v1 - v2) <= 1e-8)
      v1
    })
  }
}

class RecurrentDecoderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val recDecoder = RecurrentDecoder[Float](5).
      add(ConvLSTMPeephole[Float](7, 7, 3, 3, 1))
    val input = Tensor[Float](4, 7, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(recDecoder, input)
  }
}
