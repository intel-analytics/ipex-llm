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
import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.TorchObject.TYPE_DOUBLE_TENSOR
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class MultiRNNCellSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A MultiRNNCell " should "work in BatchMode" in {
    val hiddenSize = 5
    val inputSize = 5
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiRNNCell[Double](cells)))

    val input = Tensor[Double](batchSize, inputSize, 10, 10).rand
    val output = model.forward(input).toTensor[Double]
    for (i <- 1 to 3) {
      val output = model.forward(input)
      model.backward(input, output)
    }
  }

//  "A MultiRNNCell " should "generate correct output with convlstm" in {
//    val hiddenSize = 7
//    val inputSize = 7
//    val seqLength = 3
//    val batchSize = 2
//    val kernalW = 3
//    val kernalH = 3
//    val rec = Recurrent[Double]()
//    val cells = Array(ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1), ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Sequential[Double]()
//      .add(rec
//        .add(MultiRNNCell[Double](cells)))
//    val weights = model.getParameters()._1.clone()
//
//    val input = Tensor[Double](batchSize, seqLength, inputSize, 3, 3).rand
//    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize, 3, 3).rand
//    val output = model.forward(input).toTensor[Double]
//    val gradInput = model.backward(input, gradOutput).toTensor[Double]
//
//    val model2 = Sequential[Double]()
//      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
//        inputSize,
//        hiddenSize,
//        kernalW, kernalH,
//        1)))
//      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
//        inputSize,
//        hiddenSize,
//        kernalW, kernalH,
//        1)))
//    model2.getParameters()._1.copy(weights)
//
//    val output2 = model2.forward(input).toTensor[Double]
//    val gradInput2 = model2.backward(input, gradOutput).toTensor[Double]
//
//    output.map(output2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//
//    gradInput.map(gradInput2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//  }

//  "A MultiCell " should "generate correct output with lstm" in {
//    val hiddenSize = 10
//    val inputSize = 10
//    val seqLength = 5
//    val batchSize = 2
//    val rec = Recurrent[Double]()
//    val cells = Array(LSTM[Double](
//      inputSize,
//      hiddenSize),
//      LSTM[Double](
//        inputSize,
//        hiddenSize)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Sequential[Double]()
//      .add(rec
//        .add(MultiRNNCell[Double](cells)))
//    val weights = model.getParameters()._1.clone()
//
//    val input = Tensor[Double](batchSize, seqLength, inputSize).rand
//    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize).rand
//    val output = model.forward(input).toTensor[Double]
//    val gradInput = model.backward(input, gradOutput).toTensor[Double]
//    val gradient = model.getParameters()._2
//
//    val model2 = Sequential[Double]()
//      .add(Recurrent[Double]().add(LSTM[Double](
//        inputSize,
//        hiddenSize)))
//      .add(Recurrent[Double]().add(LSTM[Double](
//        inputSize,
//        hiddenSize)))
//    model2.getParameters()._1.copy(weights)
//
//    val output2 = model2.forward(input).toTensor[Double]
//    val gradInput2 = model2.backward(input, gradOutput).toTensor[Double]
//    val gradient2 = model2.getParameters()._2
//
//    output.map(output2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//
//    gradInput.map(gradInput2, (v1, v2) => {
//      assert(abs(v1 - v2) < 1e-6)
//      v1
//    })
//
//    require(gradient.almostEqual(gradient2, 1e-8) == true)
//  }

  "A MultiRNNCell " should "generate correct output with convlstm RecurrentDecoder" in {
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 3
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiRNNCell[Double](cells)))
    val weights = model.getParameters()._1.clone()

    val input = Tensor[Double](batchSize, inputSize, 3, 3).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize, 3, 3).rand
    val output = model.forward(input).toTensor[Double]
    val gradInput = model.backward(input, gradOutput).toTensor[Double]

    val input2 = Tensor[Double](Array(batchSize, seqLength, inputSize, 3, 3))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val model2 = Sequential[Double]()
      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
        inputSize,
        hiddenSize,
        kernalW, kernalH,
        1)))
      .add(Recurrent[Double]().add(ConvLSTMPeephole[Double](
        inputSize,
        hiddenSize,
        kernalW, kernalH,
        1)))
    model2.getParameters()._1.copy(weights)

    val output2 = model2.forward(input2).toTensor[Double]

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MultiRNCell backward" should "work with ConvLSTMPeepwhole RecurrentDecoder" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 3
    val inputSize = 3
    val seqLength = 2
    val seed = 100
    val kernalW = 3
    val kernalH = 3
    val batchSize = 2

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, inputSize, 3, 3).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3).rand
    val rec = RecurrentDecoder(seqLength)
    val cells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]
    val model = rec
      .add(MultiRNNCell(cells))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2.clone()

    val input2 = input.clone()
    input2.resize(batchSize, 1, inputSize, 3, 3)
    val model2 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    model2.getParameters()._1.copy(weights.narrow(1, 1, weights.nElement()/2))
    model2.zeroGradParameters()
    val model4 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    model4.getParameters()._1
      .copy(weights.narrow(1, weights.nElement()/2 + 1, weights.nElement()/2))
    model4.zeroGradParameters()

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

    val model5 = ConvLSTMPeephole(inputSize, hiddenSize, 3, 3, 1)
    i = 0
    while (i < model5.parameters()._1.length) {
      model5.parameters()._1(i).set(model4.parameters()._1(i))
      i += 1
    }
    i = 0
    while (i < model5.parameters()._2.length) {
      model5.parameters()._2(i).set(model4.parameters()._2(i))
      i += 1
    }

    val state = T(Tensor[Double](batchSize, hiddenSize, 3, 3),
      Tensor[Double](batchSize, hiddenSize, 3, 3))
    val state2 = T(Tensor[Double](batchSize, hiddenSize, 3, 3),
      Tensor[Double](batchSize, hiddenSize, 3, 3))
    val output2 = model2.forward(T(input, state))
    val output4 = model4.forward(T(output2(1), state2))

    val input3 = T()
    input3(1) = output4(1)
    input3(2) = output2(2)
    val output3 = model3.forward(input3)
    val input5 = T()
    input5(1) = output3(1)
    input5(2) = output4(2)
    val output5 = model5.forward(input5)

    val gradState = T(Tensor[Double](batchSize, hiddenSize, 3, 3),
      Tensor[Double](batchSize, hiddenSize, 3, 3))
    val gradState2 = T(Tensor[Double](batchSize, hiddenSize, 3, 3),
      Tensor[Double](batchSize, hiddenSize, 3, 3))
    val gradOutput5 = gradOutput.select(2, 2)
    val gradInput5 = model5.backward(input5, T(gradOutput5, gradState))

    val gradInput3 = model3.backward(input3, T(gradInput5(1), gradState2))
    val tmp_gradInput = gradInput3.clone
    tmp_gradInput(1) = gradOutput.select(2, 1).add(gradInput3.toTable[Tensor[Double]](1))
    tmp_gradInput(2) = gradInput5(2)
    val gradInput4 = model4.backward(T(output2(1), state2), tmp_gradInput)
    val gradOutput2 = T()
    gradOutput2(1) = gradInput4(1)
    gradOutput2(2) = gradInput3(2)
    val gradInput2 = model2.backward(T(input, state), gradOutput2)

    val finalOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3)
    finalOutput.narrow(2, 1, 1).copy(output4.toTable[Tensor[Double]](1))
    finalOutput.narrow(2, 2, 1).copy(output5.toTable[Tensor[Double]](1))
    require(output.almostEqual(finalOutput, 1e-8) == true)

    require(gradient.narrow(1, 1, gradient.nElement()/2)
      .almostEqual(model2.getParameters()._2, 1e-8) == true)
    require(gradient.narrow(1, gradient.nElement()/2 + 1, gradient.nElement()/2)
      .almostEqual(model4.getParameters()._2, 1e-8) == true)

    val newGradInput = Tensor[Double](batchSize, seqLength, hiddenSize, 3, 3)
    newGradInput.narrow(2, 1, 1).copy(gradInput2.toTable[Tensor[Double]](1))
    newGradInput.narrow(2, 2, 1).copy(gradInput3.toTable[Tensor[Double]](1))
    require(gradInput.almostEqual(newGradInput, 1e-8) == true)
  }

  "A MultiRNNCell " should "generate correct output with lstm RecurrentDecoder" in {
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 3
    val batchSize = 2
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(LSTM[Double](
      inputSize,
      hiddenSize), LSTM[Double](
      inputSize,
      hiddenSize)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiRNNCell[Double](cells)))
    val weights = model.getParameters()._1.clone()

    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize).rand
    val output = model.forward(input).toTensor[Double]
    val gradInput = model.backward(input, gradOutput).toTensor[Double]

    val input2 = Tensor[Double](Array(batchSize, seqLength, inputSize))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val model2 = Sequential[Double]()
      .add(Recurrent[Double]().add(LSTM[Double](
        inputSize,
        hiddenSize)))
      .add(Recurrent[Double]().add(LSTM[Double](
        inputSize,
        hiddenSize)))
    model2.getParameters()._1.copy(weights)

    val output2 = model2.forward(input2).toTensor[Double]

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })
  }

  "A MultiRNCell backward" should "work with lstm RecurrentDecoder" in {
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
    val cells = Array(LSTM[Double](
      inputSize,
      hiddenSize), LSTM[Double](
      inputSize,
      hiddenSize)).asInstanceOf[Array[Cell[Double]]]
    val model = rec
      .add(MultiRNNCell(cells))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2.clone()

    val input2 = input.clone()
    input2.resize(batchSize, 1, inputSize)
    val model2 = LSTM(inputSize, hiddenSize)
    model2.includePreTopology = true
    model2.getParameters()._1.copy(weights.narrow(1, 1, weights.nElement()/2))
    model2.zeroGradParameters()
    val model4 = LSTM(inputSize, hiddenSize)
    model4.includePreTopology = true
    model4.getParameters()._1
      .copy(weights.narrow(1, weights.nElement()/2 + 1, weights.nElement()/2))
    model4.zeroGradParameters()

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

    val model5 = LSTM(inputSize, hiddenSize)
    model5.includePreTopology = true
    i = 0
    while (i < model5.parameters()._1.length) {
      model5.parameters()._1(i).set(model4.parameters()._1(i))
      i += 1
    }
    i = 0
    while (i < model5.parameters()._2.length) {
      model5.parameters()._2(i).set(model4.parameters()._2(i))
      i += 1
    }

    val state = T(Tensor[Double](batchSize, hiddenSize),
      Tensor[Double](batchSize, hiddenSize))
    val state2 = T(Tensor[Double](batchSize, hiddenSize),
      Tensor[Double](batchSize, hiddenSize))
    val output2 = model2.forward(T(input, state))
    val output4 = model4.forward(T(output2(1), state2))

    val input3 = T()
    input3(1) = output4(1)
    input3(2) = output2(2)
    val output3 = model3.forward(input3)
    val input5 = T()
    input5(1) = output3(1)
    input5(2) = output4(2)
    val output5 = model5.forward(input5)

    val gradState = T(Tensor[Double](batchSize, hiddenSize),
      Tensor[Double](batchSize, hiddenSize))
    val gradState2 = T(Tensor[Double](batchSize, hiddenSize),
      Tensor[Double](batchSize, hiddenSize))
    val gradOutput5 = gradOutput.select(2, 2)
    val gradInput5 = model5.backward(input5, T(gradOutput5, gradState))

    val gradInput3 = model3.backward(input3, T(gradInput5(1), gradState2))
    val tmp_gradInput = gradInput3.clone
    tmp_gradInput(1) = gradOutput.select(2, 1).add(gradInput3.toTable[Tensor[Double]](1))
    tmp_gradInput(2) = gradInput5(2)
    val gradInput4 = model4.backward(T(output2(1), state2), tmp_gradInput)
    val gradOutput2 = T()
    gradOutput2(1) = gradInput4(1)
    gradOutput2(2) = gradInput3(2)
    val gradInput2 = model2.backward(T(input, state), gradOutput2)

    val finalOutput = Tensor[Double](batchSize, seqLength, hiddenSize)
    finalOutput.narrow(2, 1, 1).copy(output4.toTable[Tensor[Double]](1))
    finalOutput.narrow(2, 2, 1).copy(output5.toTable[Tensor[Double]](1))
    require(output.almostEqual(finalOutput, 1e-8) == true)

    require(gradient.narrow(1, 1, gradient.nElement()/2)
      .almostEqual(model2.getParameters()._2, 1e-8) == true)
    require(gradient.narrow(1, gradient.nElement()/2 + 1, gradient.nElement()/2)
      .almostEqual(model4.getParameters()._2, 1e-8) == true)

    val newGradInput = Tensor[Double](batchSize, seqLength, hiddenSize)
    newGradInput.narrow(2, 1, 1).copy(gradInput2.toTable[Tensor[Double]](1))
    newGradInput.narrow(2, 2, 1).copy(gradInput3.toTable[Tensor[Double]](1))
    require(gradInput.almostEqual(newGradInput, 1e-8) == true)
  }

  "A MultiRNCell updateGradInput/acc" should "work with lstm RecurrentDecoder" in {
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
    val cells = Array(LSTM[Double](
      inputSize,
      hiddenSize), LSTM[Double](
      inputSize,
      hiddenSize)).asInstanceOf[Array[Cell[Double]]]
    val model = rec
      .add(MultiRNNCell(cells))

    val rec2 = RecurrentDecoder(seqLength)
    val cells2 = Array(LSTM[Double](
      inputSize,
      hiddenSize), LSTM[Double](
      inputSize,
      hiddenSize)).asInstanceOf[Array[Cell[Double]]]
    val model2 = rec2
      .add(MultiRNNCell(cells2))

    val weights = model.getParameters()._1.clone()
    model.zeroGradParameters()
    model2.getParameters()._1.copy(weights)
    model2.zeroGradParameters()

    val output = model.forward(input).toTensor
    val gradInput = model.backward(input, gradOutput).toTensor
    val gradient = model.getParameters()._2

    val output2 = model2.forward(input).toTensor
    val gradInput2 = model2.updateGradInput(input, gradOutput).toTensor
    model2.accGradParameters(input, gradOutput)
    val gradient2 = model2.getParameters()._2

    require(output.almostEqual(output2, 1e-8) == true)
    require(gradient.almostEqual(gradient2, 1e-8) == true)
    require(gradInput.almostEqual(gradInput2, 1e-8) == true)
  }

  "A MultiRNNCell " should "work with set/getHiddenState" in {
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 3
    val batchSize = 2
    val rec = RecurrentDecoder[Double](seqLength)
    val cells = Array(LSTM[Double](
      inputSize,
      hiddenSize), LSTM[Double](
      inputSize,
      hiddenSize)).asInstanceOf[Array[Cell[Double]]]

    val model = Sequential[Double]()
      .add(rec
        .add(MultiRNNCell[Double](cells)))
    val weights = model.getParameters()._1.clone()

    val input = Tensor[Double](batchSize, inputSize).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, inputSize).rand
    val map0 = mutable.HashMap[Any, Any]()
    val hidden0 = Tensor[Double](batchSize, hiddenSize).rand
    val state0 = T(hidden0,
      Tensor[Double](batchSize, hiddenSize).rand)
    map0.put(1, state0)
    val map1 = mutable.HashMap[Any, Any]()
    val state1 = T(Tensor[Double](batchSize, hiddenSize).rand,
      Tensor[Double](batchSize, hiddenSize).rand)
    map1.put(1, state1)
    val map = mutable.HashMap[Any, Any]()
    map.put(1, state0)
    map.put(2, state1)
    val initStates = new Table(map)
    val initStates_0 = new Table(map0)
    val initStates_1 = new Table(map1)
    rec.setHiddenState(initStates)
    val output = model.forward(input).toTensor[Double]
    val gradInput = model.backward(input, gradOutput).toTensor[Double]

    val input2 = Tensor[Double](Array(batchSize, seqLength, inputSize))
    input2.narrow(2, 1, 1).copy(input)
    input2.narrow(2, 2, seqLength-1).copy(output.narrow(2, 1, seqLength-1))
    val rec0 = Recurrent[Double]().add(LSTM[Double](
      inputSize,
      hiddenSize))
    rec0.setHiddenState(initStates_0)
    val rec1 = Recurrent[Double]().add(LSTM[Double](
      inputSize,
      hiddenSize))
    rec1.setHiddenState(initStates_1)
    val model2 = Sequential[Double]()
      .add(rec0)
      .add(rec1)
    model2.getParameters()._1.copy(weights)

    val output2 = model2.forward(input2).toTensor[Double]

    output.map(output2, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6)
      v1
    })

    val state_decoder0 = rec.getHiddenState().toTable[Table](1).getState()
    val state_decoder1 = rec.getHiddenState().toTable[Table](2).getState()
    val stateGet0 = rec0.getHiddenState().toTable.getState()
    val stateGet1 = rec1.getHiddenState().toTable.getState()
    for (k <- state_decoder0.keys) {
      val t1 = state_decoder0(k).asInstanceOf[Tensor[Double]]
      val t2 = stateGet0(k).asInstanceOf[Tensor[Double]]
      t1.map(t2, (v1, v2) => {
        assert(abs(v1 - v2) <= 1e-8)
        v1
      })
    }

    for (k <- state_decoder1.keys) {
      val t1 = state_decoder1(k).asInstanceOf[Tensor[Double]]
      val t2 = stateGet1(k).asInstanceOf[Tensor[Double]]
      t1.map(t2, (v1, v2) => {
        assert(abs(v1 - v2) <= 1e-8)
        v1
      })
    }

    // init states shoule remain unchanged
    initStates.get(1).get.asInstanceOf[Table].get(1).get
      .asInstanceOf[Tensor[Double]].map(hidden0, (v1, v2) => {
      assert(v1 == v2)
      v1
    })

    rec.setHiddenState(rec.getHiddenState())
    model.forward(input)
  }
}

class MultiRNNCellSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val hiddenSize = 5
    val inputSize = 5
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = RecurrentDecoder[Float](seqLength)
    val cells = Array(ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Float]]]

    val multiRNNCell = MultiRNNCell[Float](cells)

    val model = Sequential[Float]()
      .add(rec
        .add(multiRNNCell)).setName("multiRNNCell")

    val input = Tensor[Float](batchSize, inputSize, 10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input, multiRNNCell.getClass)
  }
}
