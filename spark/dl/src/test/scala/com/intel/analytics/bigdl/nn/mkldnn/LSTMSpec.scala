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


package com.intel.analytics.bigdl.nn.mkldnn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

class LSTMSpec extends FlatSpec with Matchers{
  "LSTM UnidirectionalInference updateOutput" should "work correctly" in {
    val seqLength = 7
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 1,
        inputSize, lstm_n_gates, hiddenSize)).rand()

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 1,
      hiddenSize, lstm_n_gates, hiddenSize)).rand()

    var initBias = Tensor[Float](
      Array(common_n_layers, 1,
        lstm_n_gates, hiddenSize)).rand()

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Uni Left2Right \n" + output1)

    direction = Direction.UnidirectionalRight2Left
    val lstm2 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm2.setRuntime(new MklDnnRuntime)
    lstm2.initMemoryDescs(Array(inputFormat))
    lstm2.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output2 = lstm2.forward(input)
    println("DNN output Uni Right2Left \n" + output2)

    input = input.transpose(1, 2)
    initWeight = initWeight.resize(Array(inputSize, lstm_n_gates, hiddenSize))
                           .transpose(1, 2)
    initWeightIter = initWeightIter.resize(Array(hiddenSize, lstm_n_gates, hiddenSize))
                                   .transpose(1, 2)
    initBias = initBias.resize(Array(lstm_n_gates, hiddenSize))
    val nn_model = nn.Recurrent().add(nn.LSTM2(inputSize, hiddenSize))

    /**
      * MKLDNN Gate 1 -> nn/LSTM2 Gate 1
      * MKLDNN Gate 2 -> nn/LSTM2 Gate 3
      * MKLDNN Gate 3 -> nn/LSTM2 Gate 2
      * MKLDNN Gate 4 -> nn/LSTM2 Gate 4
      *
      * uniParams(0), uniParams(1) -> nn/LSTM2 i2g4
      * uniParams(2), uniParams(3) -> nn/LSTM2 i2g3
      * uniParams(4), uniParams(5) -> nn/LSTM2 i2g2
      * uniParams(6), uniParams(7) -> nn/LSTM2 i2g1
      * uniParams(8), uniParams(9) -> nn/LSTM2 h2g4
      * uniParams(10), uniParams(11) -> nn/LSTM2 h2g3
      * uniParams(12), uniParams(13) -> nn/LSTM2 h2g2
      * uniParams(14), uniParams(15) -> nn/LSTM2 h2g2
      */
    val uniParams = nn_model.parameters()._1
    uniParams(0).copy(initWeight(4).transpose(1, 2))
    uniParams(1).copy(initBias(4))
    uniParams(2).copy(initWeight(2).transpose(1, 2))
    uniParams(3).copy(initBias(2))
    uniParams(4).copy(initWeight(3).transpose(1, 2))
    uniParams(5).copy(initBias(3))
    uniParams(6).copy(initWeight(1).transpose(1, 2))
    uniParams(7).copy(initBias(1))
    uniParams(8).copy(initWeightIter(4).transpose(1, 2))
    uniParams(9).copy(Tensor[Float](initBias(4)).zero())
    uniParams(10).copy(initWeightIter(2).transpose(1, 2))
    uniParams(11).copy(Tensor[Float](initBias(2)).zero())
    uniParams(12).copy(initWeightIter(3).transpose(1, 2))
    uniParams(13).copy(Tensor[Float](initBias(3)).zero())
    uniParams(14).copy(initWeightIter(1).transpose(1, 2))
    uniParams(15).copy(Tensor[Float](initBias(1)).zero())

    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Uni Left2Right \n" + nn_output)

    Equivalent.nearequals(Tools.dense(output1).asInstanceOf[Tensor[Float]],
      nn_output) should be(true)

    /**
      * nn/LSTM2 Right2Left
      */
    val reverse = nn.Reverse(2)
    input = reverse.forward(input)

    var nn_output2 = nn_model.forward(input)
    nn_output2 = reverse.forward(nn_output2).toTensor.transpose(1, 2)
    println("NN output Uni Right2Left \n" + nn_output2)

    Equivalent.nearequals(Tools.dense(output2).asInstanceOf[Tensor[Float]],
      nn_output2) should be(true)
  }

  "LSTM BidirectionalConcatInference updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.BidirectionalConcat

    val common_n_layers = 1
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 2,
        inputSize, lstm_n_gates, hiddenSize)).rand()

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 2,
        hiddenSize, lstm_n_gates, hiddenSize)).rand()

    var initBias = Tensor[Float](
      Array(common_n_layers, 2,
        lstm_n_gates, hiddenSize)).rand()

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Bi Concat \n" + output1)


    input = input.transpose(1, 2)
    initWeight = initWeight.resize(Array(2, inputSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3)
    initWeightIter = initWeightIter.resize(Array(2, hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3)
    initBias = initBias.resize(Array(2, lstm_n_gates, hiddenSize))

    val nn_model = nn.BiRecurrent[Float](nn.JoinTable[Float](3, 0)
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM2(inputSize, hiddenSize))

    /**
      * biParams(0 - 15) and (16 - 31) are for the two directions respectively
      */
    val biParams = nn_model.parameters()._1
    biParams(0).copy(initWeight(1)(4).transpose(1, 2))
    biParams(1).copy(initBias(1)(4))
    biParams(2).copy(initWeight(1)(2).transpose(1, 2))
    biParams(3).copy(initBias(1)(2))
    biParams(4).copy(initWeight(1)(3).transpose(1, 2))
    biParams(5).copy(initBias(1)(3))
    biParams(6).copy(initWeight(1)(1).transpose(1, 2))
    biParams(7).copy(initBias(1)(1))
    biParams(8).copy(initWeightIter(1)(4).transpose(1, 2))
    biParams(9).copy(Tensor[Float](initBias(1)(4)).zero())
    biParams(10).copy(initWeightIter(1)(2).transpose(1, 2))
    biParams(11).copy(Tensor[Float](initBias(1)(2)).zero())
    biParams(12).copy(initWeightIter(1)(3).transpose(1, 2))
    biParams(13).copy(Tensor[Float](initBias(1)(3)).zero())
    biParams(14).copy(initWeightIter(1)(1).transpose(1, 2))
    biParams(15).copy(Tensor[Float](initBias(1)(1)).zero())

    biParams(16).copy(initWeight(2)(4).transpose(1, 2))
    biParams(17).copy(initBias(2)(4))
    biParams(18).copy(initWeight(2)(2).transpose(1, 2))
    biParams(19).copy(initBias(2)(2))
    biParams(20).copy(initWeight(2)(3).transpose(1, 2))
    biParams(21).copy(initBias(2)(3))
    biParams(22).copy(initWeight(2)(1).transpose(1, 2))
    biParams(23).copy(initBias(2)(1))
    biParams(24).copy(initWeightIter(2)(4).transpose(1, 2))
    biParams(25).copy(Tensor[Float](initBias(2)(4)).zero())
    biParams(26).copy(initWeightIter(2)(2).transpose(1, 2))
    biParams(27).copy(Tensor[Float](initBias(2)(2)).zero())
    biParams(28).copy(initWeightIter(2)(3).transpose(1, 2))
    biParams(29).copy(Tensor[Float](initBias(2)(3)).zero())
    biParams(30).copy(initWeightIter(2)(1).transpose(1, 2))
    biParams(31).copy(Tensor[Float](initBias(2)(1)).zero())

    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Bi Concat \n" + nn_output)

    Equivalent.nearequals(Tools.dense(output1).asInstanceOf[Tensor[Float]],
      nn_output) should be(true)
  }

  "LSTM BidirectionalSumInference updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.BidirectionalSum

    val common_n_layers = 1
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 2,
        inputSize, lstm_n_gates, hiddenSize)).rand()

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 2,
        hiddenSize, lstm_n_gates, hiddenSize)).rand()

    var initBias = Tensor[Float](
      Array(common_n_layers, 2,
        lstm_n_gates, hiddenSize)).rand()

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)

    var output1 = lstm1.forward(input)
    println("DNN output Bi Sum \n" + output1)

    input = input.transpose(1, 2)
    initWeight = initWeight.resize(Array(2, inputSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3)
    initWeightIter = initWeightIter.resize(Array(2, hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3)
    initBias = initBias.resize(Array(2, lstm_n_gates, hiddenSize))

    val nn_model = nn.BiRecurrent[Float](nn.CAddTable())
      .add(nn.LSTM2(inputSize, hiddenSize))

    /**
      * biParams(0 - 15) and (16 - 31) are for the two directions respectively
      */
    val biParams = nn_model.parameters()._1
    biParams(0).copy(initWeight(1)(4).transpose(1, 2))
    biParams(1).copy(initBias(1)(4))
    biParams(2).copy(initWeight(1)(2).transpose(1, 2))
    biParams(3).copy(initBias(1)(2))
    biParams(4).copy(initWeight(1)(3).transpose(1, 2))
    biParams(5).copy(initBias(1)(3))
    biParams(6).copy(initWeight(1)(1).transpose(1, 2))
    biParams(7).copy(initBias(1)(1))
    biParams(8).copy(initWeightIter(1)(4).transpose(1, 2))
    biParams(9).copy(Tensor[Float](initBias(1)(4)).zero())
    biParams(10).copy(initWeightIter(1)(2).transpose(1, 2))
    biParams(11).copy(Tensor[Float](initBias(1)(2)).zero())
    biParams(12).copy(initWeightIter(1)(3).transpose(1, 2))
    biParams(13).copy(Tensor[Float](initBias(1)(3)).zero())
    biParams(14).copy(initWeightIter(1)(1).transpose(1, 2))
    biParams(15).copy(Tensor[Float](initBias(1)(1)).zero())

    biParams(16).copy(initWeight(2)(4).transpose(1, 2))
    biParams(17).copy(initBias(2)(4))
    biParams(18).copy(initWeight(2)(2).transpose(1, 2))
    biParams(19).copy(initBias(2)(2))
    biParams(20).copy(initWeight(2)(3).transpose(1, 2))
    biParams(21).copy(initBias(2)(3))
    biParams(22).copy(initWeight(2)(1).transpose(1, 2))
    biParams(23).copy(initBias(2)(1))
    biParams(24).copy(initWeightIter(2)(4).transpose(1, 2))
    biParams(25).copy(Tensor[Float](initBias(2)(4)).zero())
    biParams(26).copy(initWeightIter(2)(2).transpose(1, 2))
    biParams(27).copy(Tensor[Float](initBias(2)(2)).zero())
    biParams(28).copy(initWeightIter(2)(3).transpose(1, 2))
    biParams(29).copy(Tensor[Float](initBias(2)(3)).zero())
    biParams(30).copy(initWeightIter(2)(1).transpose(1, 2))
    biParams(31).copy(Tensor[Float](initBias(2)(1)).zero())

    val nn_output = nn_model.forward(input).toTensor.transpose(1, 2)
    println("NN output Bi Sum \n" + nn_output)

    Equivalent.nearequals(Tools.dense(output1).asInstanceOf[Tensor[Float]],
      nn_output) should be(true)
  }
}
