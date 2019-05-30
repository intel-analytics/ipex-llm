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

class RNNSpec extends FlatSpec with Matchers{
  "LSTM UnidirectionalInference updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 1,
        inputSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 1,
      hiddenSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 1,
        lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    val lstm1 = RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output LSTM Uni Left2Right \n" + output1)

    direction = Direction.UnidirectionalRight2Left
    val lstm2 = RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm2.setRuntime(new MklDnnRuntime)
    lstm2.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output2 = lstm2.forward(input)
    println("DNN output LSTM Uni Right2Left \n" + output2)

    var inputt = input.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(inputSize, lstm_n_gates, hiddenSize))
      .transpose(1, 2).transpose(2, 3)
    initWeightIter = initWeightIter.resize(Array(hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(1, 2).transpose(2, 3)
    initBias = initBias.resize(Array(lstm_n_gates, hiddenSize))

    /**
      * MKLDNN Gate 1 -> nn/LSTM Gate 1
      * MKLDNN Gate 2 -> nn/LSTM Gate 3
      * MKLDNN Gate 3 -> nn/LSTM Gate 2
      * MKLDNN Gate 4 -> nn/LSTM Gate 4
      *
      * uniParams(0) -> input weights
      * uniParams(1) -> bias
      * uniParams(2) -> hidden weights
      */

    var initWeight0 = Tensor[Float](Array(hiddenSize * lstm_n_gates, inputSize))
    var initWeightIter0 = Tensor[Float](Array(hiddenSize * lstm_n_gates, hiddenSize))
    var initBias0 = Tensor[Float](Array(lstm_n_gates * hiddenSize))

    val concat = nn.JoinTable(1, 4)
    initWeight0 = concat.forward(T(initWeight(1), initWeight(3),
      initWeight(2), initWeight(4))).asInstanceOf[Tensor[Float]].clone()
    initWeightIter0 = concat.forward(T(initWeightIter(1), initWeightIter(3),
      initWeightIter(2), initWeightIter(4))).asInstanceOf[Tensor[Float]].clone()
    initBias0 = concat.forward(T(initBias(1), initBias(3), initBias(2), initBias(4)))
      .asInstanceOf[Tensor[Float]].clone()

    val nn_model = nn.Recurrent().add(nn.LSTM(inputSize, hiddenSize))

    val uniParams = nn_model.parameters()._1
    initWeight0 = initWeight0.resizeAs(uniParams(0))
    initBias0 = initBias0.resizeAs(uniParams(1))
    initWeightIter0 = initWeightIter0.resizeAs(uniParams(2))

    uniParams(0).copy(initWeight0)
    uniParams(1).copy(initBias0)
    uniParams(2).copy(initWeightIter0)

    val nn_output = nn_model.forward(inputt).toTensor.transpose(1, 2)
    println("NN output LSTM Uni Left2Right \n" + nn_output)

    Equivalent.nearequals(Tools.dense(output1).asInstanceOf[Tensor[Float]],
      nn_output) should be(true)

    /**
      * nn/LSTM Right2Left
      */
    val reverse = nn.Reverse(2)
    inputt = reverse.forward(inputt)

    var nn_output2 = nn_model.forward(inputt)
    nn_output2 = reverse.forward(nn_output2).toTensor.transpose(1, 2)
    println("NN output LSTM Uni Right2Left \n" + nn_output2)
    println("==================================================================== \n\n\n")

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

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 2,
        inputSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 2,
        hiddenSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 2,
        lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    val lstm1 = RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output LSTM Bi Concat \n" + output1)

    val inputt = input.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(2, inputSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3).transpose(3, 4)
    initWeightIter = initWeightIter.resize(Array(2, hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3).transpose(3, 4)
    initBias = initBias.resize(Array(2, lstm_n_gates, hiddenSize))

    var initWeight0 = Tensor[Float](Array(2, hiddenSize * lstm_n_gates, inputSize))
    var initWeightIter0 = Tensor[Float](Array(2, hiddenSize * lstm_n_gates, hiddenSize))
    var initBias0 = Tensor[Float](Array(2, lstm_n_gates * hiddenSize))

    val concat = nn.JoinTable(1, 4)
    initWeight0(1) = concat.forward(T(initWeight(1)(1), initWeight(1)(3),
      initWeight(1)(2), initWeight(1)(4))).asInstanceOf[Tensor[Float]].clone()
    initWeightIter0(1) = concat.forward(T(initWeightIter(1)(1), initWeightIter(1)(3),
      initWeightIter(1)(2), initWeightIter(1)(4))).asInstanceOf[Tensor[Float]].clone()
    initBias0(1) = concat.forward(T(initBias(1)(1), initBias(1)(3),
      initBias(1)(2), initBias(1)(4))).asInstanceOf[Tensor[Float]].clone()

    initWeight0(2) = concat.forward(T(initWeight(2)(1), initWeight(2)(3),
      initWeight(2)(2), initWeight(2)(4))).asInstanceOf[Tensor[Float]].clone()
    initWeightIter0(2) = concat.forward(T(initWeightIter(2)(1), initWeightIter(2)(3),
      initWeightIter(2)(2), initWeightIter(2)(4))).asInstanceOf[Tensor[Float]].clone()
    initBias0(2) = concat.forward(T(initBias(2)(1), initBias(2)(3),
      initBias(2)(2), initBias(2)(4))).asInstanceOf[Tensor[Float]].clone()

    val nn_model = nn.BiRecurrent[Float](nn.JoinTable[Float](3, 0)
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM(inputSize, hiddenSize))

    /**
      * biParams(0 - 2) and (3 - 5) are for the two directions respectively
      *
      * MKLDNN Gate 1 -> nn/LSTM Gate 1
      * MKLDNN Gate 2 -> nn/LSTM Gate 3
      * MKLDNN Gate 3 -> nn/LSTM Gate 2
      * MKLDNN Gate 4 -> nn/LSTM Gate 4
      *
      * biParams(0) -> input weights
      * biParams(1) -> bias
      * biParams(2) -> hidden weights
      * biParams(3) -> input weights
      * biParams(4) -> bias
      * biParams(5) -> hidden weights
      */

    val biParams = nn_model.parameters()._1
    initWeight0(1).resizeAs(biParams(0))
    initBias0(1).resizeAs(biParams(1))
    initWeightIter0(1).resizeAs(biParams(2))
    initWeight0(2).resizeAs(biParams(3))
    initBias0(2).resizeAs(biParams(4))
    initWeightIter0(2).resizeAs(biParams(5))

    biParams(0).copy(initWeight0(1))
    biParams(1).copy(initBias0(1))
    biParams(2).copy(initWeightIter0(1))
    biParams(3).copy(initWeight0(2))
    biParams(4).copy(initBias0(2))
    biParams(5).copy(initWeightIter0(2))

    val nn_output = nn_model.forward(inputt).toTensor.transpose(1, 2)
    println("NN output LSTM Bi Concat \n" + nn_output)
    println("==================================================================== \n\n\n")

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

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    var input = Tensor(Array(seqLength, batchSize, inputSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 2,
        inputSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 2,
        hiddenSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 2,
        lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    val lstm1 = RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output LSTM Bi Sum \n" + output1)

    val inputt = input.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(2, inputSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3).transpose(3, 4)
    initWeightIter = initWeightIter.resize(Array(2, hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(2, 3).transpose(3, 4)
    initBias = initBias.resize(Array(2, lstm_n_gates, hiddenSize))

    var initWeight0 = Tensor[Float](Array(2, hiddenSize * lstm_n_gates, inputSize))
    var initWeightIter0 = Tensor[Float](Array(2, hiddenSize * lstm_n_gates, hiddenSize))
    var initBias0 = Tensor[Float](Array(2, lstm_n_gates * hiddenSize))

    val concat = nn.JoinTable(1, 4)
    initWeight0(1) = concat.forward(T(initWeight(1)(1), initWeight(1)(3),
      initWeight(1)(2), initWeight(1)(4))).asInstanceOf[Tensor[Float]].clone()
    initWeightIter0(1) = concat.forward(T(initWeightIter(1)(1), initWeightIter(1)(3),
      initWeightIter(1)(2), initWeightIter(1)(4))).asInstanceOf[Tensor[Float]].clone()
    initBias0(1) = concat.forward(T(initBias(1)(1), initBias(1)(3),
      initBias(1)(2), initBias(1)(4))).asInstanceOf[Tensor[Float]].clone()

    initWeight0(2) = concat.forward(T(initWeight(2)(1), initWeight(2)(3),
      initWeight(2)(2), initWeight(2)(4))).asInstanceOf[Tensor[Float]].clone()
    initWeightIter0(2) = concat.forward(T(initWeightIter(2)(1), initWeightIter(2)(3),
      initWeightIter(2)(2), initWeightIter(2)(4))).asInstanceOf[Tensor[Float]].clone()
    initBias0(2) = concat.forward(T(initBias(2)(1), initBias(2)(3),
      initBias(2)(2), initBias(2)(4))).asInstanceOf[Tensor[Float]].clone()

    val nn_model = nn.BiRecurrent[Float](nn.CAddTable()
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM(inputSize, hiddenSize))

    /**
      * biParams(0 - 2) and (3 - 5) are for the two directions respectively
      *
      * MKLDNN Gate 1 -> nn/LSTM Gate 1
      * MKLDNN Gate 2 -> nn/LSTM Gate 3
      * MKLDNN Gate 3 -> nn/LSTM Gate 2
      * MKLDNN Gate 4 -> nn/LSTM Gate 4
      *
      * biParams(0) -> input weights
      * biParams(1) -> bias
      * biParams(2) -> hidden weights
      * biParams(3) -> input weights
      * biParams(4) -> bias
      * biParams(5) -> hidden weights
      */

    val biParams = nn_model.parameters()._1
    initWeight0(1).resizeAs(biParams(0))
    initBias0(1).resizeAs(biParams(1))
    initWeightIter0(1).resizeAs(biParams(2))
    initWeight0(2).resizeAs(biParams(3))
    initBias0(2).resizeAs(biParams(4))
    initWeightIter0(2).resizeAs(biParams(5))

    biParams(0).copy(initWeight0(1))
    biParams(1).copy(initBias0(1))
    biParams(2).copy(initWeightIter0(1))
    biParams(3).copy(initWeight0(2))
    biParams(4).copy(initBias0(2))
    biParams(5).copy(initWeightIter0(2))

    val nn_output = nn_model.forward(inputt).toTensor.transpose(1, 2)
    println("NN output LSTM Bi Sum \n" + nn_output)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(output1).asInstanceOf[Tensor[Float]],
      nn_output) should be(true)
  }

  "LSTM UnidirectionalInference Multilayers updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val commonSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 3
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, commonSize), Memory.Format.tnc)
    var input = Tensor(Array(seqLength, batchSize, commonSize)).rand()

    var initWeight = Tensor[Float](
      Array(common_n_layers, 1,
        commonSize, lstm_n_gates, commonSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 1,
        commonSize, lstm_n_gates, commonSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 1,
        lstm_n_gates, commonSize)).rand(-1.0, 1.0)

    val lstm = RNN(AlgKind.VanillaLstm, commonSize, commonSize, f, direction
      , layers = common_n_layers,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    lstm.setRuntime(new MklDnnRuntime)
    lstm.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output = lstm.forward(input)
    println("DNN output LSTM Uni Multilayers Left2Right \n" + output)

    var inputt = input.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(common_n_layers, commonSize, lstm_n_gates, commonSize))
      .transpose(2, 3).transpose(3, 4)
    initWeightIter = initWeightIter
      .resize(Array(common_n_layers, commonSize, lstm_n_gates, commonSize))
      .transpose(2, 3).transpose(3, 4)
    initBias = initBias.resize(Array(common_n_layers, lstm_n_gates, commonSize))

    /**
      * MKLDNN Gate 1 -> nn/LSTM Gate 1
      * MKLDNN Gate 2 -> nn/LSTM Gate 3
      * MKLDNN Gate 3 -> nn/LSTM Gate 2
      * MKLDNN Gate 4 -> nn/LSTM Gate 4
      *
      * uniParams(0) -> input weights
      * uniParams(1) -> bias
      * uniParams(2) -> hidden weights
      */

    var initWeight0 = Tensor[Float](Array(common_n_layers, commonSize * lstm_n_gates, commonSize))
    var initWeightIter0 =
      Tensor[Float](Array(common_n_layers, commonSize * lstm_n_gates, commonSize))
    var initBias0 = Tensor[Float](Array(common_n_layers, lstm_n_gates * commonSize))

    val concat = nn.JoinTable(1, 4)
    for(l <- 1 to common_n_layers) {
      initWeight0(l).copy(concat.forward(T(initWeight(l)(1), initWeight(l)(3),
        initWeight(l)(2), initWeight(l)(4))).asInstanceOf[Tensor[Float]].clone())
      initWeightIter0(l).copy(concat.forward(T(initWeightIter(l)(1), initWeightIter(l)(3),
        initWeightIter(l)(2), initWeightIter(l)(4))).asInstanceOf[Tensor[Float]].clone())
      initBias0(l).copy(concat.forward(T(initBias(l)(1), initBias(l)(3),
        initBias(l)(2), initBias(l)(4)))
        .asInstanceOf[Tensor[Float]].clone())
    }

    val nn_input = nn.Input()
    var nn_lstm = nn.Recurrent().add(nn.LSTM(commonSize, commonSize)).inputs(nn_input)

    for(i <- 1 until common_n_layers) {
      nn_lstm = nn.Recurrent().add(nn.LSTM(commonSize, commonSize)).inputs(nn_lstm)
    }

    val nn_model = nn.Graph(nn_input, nn_lstm)

    val uniParams = nn_model.parameters()._1

    for(l <- 0 until common_n_layers) {
      initWeight0(l + 1) = initWeight0(l + 1).resizeAs(uniParams(3 * l))
      initBias0(l + 1) = initBias0(l + 1).resizeAs(uniParams(3 * l + 1))
      initWeightIter0(l + 1) = initWeightIter0(l + 1).resizeAs(uniParams(3 * l + 2))

      uniParams(3 * l).copy(initWeight0(l + 1))
      uniParams(3 * l + 1).copy(initBias0(l + 1))
      uniParams(3 * l + 2).copy(initWeightIter0(l + 1))
    }

    val nn_output = nn_model.forward(inputt).toTensor.transpose(1, 2)
    println("NN output LSTM Uni Multilayers Left2Right \n" + nn_output)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(output).asInstanceOf[Tensor[Float]],
      nn_output) should be(true)
  }
}
