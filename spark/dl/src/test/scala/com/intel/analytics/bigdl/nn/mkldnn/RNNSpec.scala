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
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
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

    val mkldnnLSTM1 = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias))
    mkldnnLSTM1.evaluate()
    mkldnnLSTM1.compile(InferencePhase)
    val mkldnn_output1 = mkldnnLSTM1.forward(input)
    println("MKLDNN output LSTM Uni Left2Right \n" + mkldnn_output1)

    direction = Direction.UnidirectionalRight2Left
    val mkldnnLSTM2 = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias))
    mkldnnLSTM2.evaluate()
    mkldnnLSTM2.compile(InferencePhase)
    val mkldnn_output2 = mkldnnLSTM2.forward(input)
    println("MKLDNN output LSTM Uni Right2Left \n" + mkldnn_output2)

    /**
     * Reorder to formats of BLAS.
     * The input format of MKLDNN is TNC, while that of BLAS is NTC.
     */
    var inputt = input.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(inputSize, lstm_n_gates, hiddenSize))
      .transpose(1, 2).transpose(2, 3)
    initWeightIter = initWeightIter.resize(Array(hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(1, 2).transpose(2, 3)
    initBias = initBias.resize(Array(lstm_n_gates, hiddenSize))

    /**
     * Gate order matching between MKLDNN LSTM and nn/LSTM:
     * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
     * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
     * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
     * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
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

    val blasLSTM = nn.Recurrent().add(nn.LSTM(inputSize, hiddenSize))

    val uniParams = blasLSTM.parameters()._1
    initWeight0 = initWeight0.resizeAs(uniParams(0))
    initBias0 = initBias0.resizeAs(uniParams(1))
    initWeightIter0 = initWeightIter0.resizeAs(uniParams(2))

    uniParams(0).copy(initWeight0)
    uniParams(1).copy(initBias0)
    uniParams(2).copy(initWeightIter0)

    val blas_output1 = blasLSTM.forward(inputt).toTensor.transpose(1, 2)
    println("BLAS output LSTM Uni Left2Right \n" + blas_output1)

    Equivalent.nearequals(Tools.dense(mkldnn_output1).asInstanceOf[Tensor[Float]],
      blas_output1) should be(true)

    /**
     * nn/LSTM Right2Left
     */
    val reverse = nn.Reverse(2)
    inputt = reverse.forward(inputt)

    var blas_output2 = blasLSTM.forward(inputt)
    blas_output2 = reverse.forward(blas_output2).toTensor.transpose(1, 2)
    println("BLAS output LSTM Uni Right2Left \n" + blas_output2)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(mkldnn_output2).asInstanceOf[Tensor[Float]],
      blas_output2) should be(true)
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

    val mkldnnLSTM = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
          initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias))
    mkldnnLSTM.evaluate()
    mkldnnLSTM.compile(InferencePhase)
    val mkldnn_output = mkldnnLSTM.forward(input)
    println("MKLDNN output LSTM Bi Concat \n" + mkldnn_output)

    /**
     * Reorder to formats of BLAS.
     * The input format of MKLDNN is TNC, while that of BLAS is NTC.
     */
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

    val blasLSTM = nn.BiRecurrent[Float](nn.JoinTable[Float](3, 0)
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM(inputSize, hiddenSize))

    /**
     * biParams(0 - 2) and (3 - 5) are for the two directions respectively
     *
     * Gate order matching between MKLDNN LSTM and nn/LSTM:
     * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
     * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
     * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
     * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
     *
     * biParams(0) -> input weights
     * biParams(1) -> bias
     * biParams(2) -> hidden weights
     * biParams(3) -> input weights
     * biParams(4) -> bias
     * biParams(5) -> hidden weights
     */

    val biParams = blasLSTM.parameters()._1
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

    val blas_output = blasLSTM.forward(inputt).toTensor.transpose(1, 2)
    println("BLAS output LSTM Bi Concat \n" + blas_output)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(mkldnn_output).asInstanceOf[Tensor[Float]],
      blas_output) should be(true)
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

    val mkldnnLSTM = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias))
    mkldnnLSTM.evaluate()
    mkldnnLSTM.compile(InferencePhase)
    val mkldnn_output = mkldnnLSTM.forward(input)
    println("MKLDNN output LSTM Bi Sum \n" + mkldnn_output)

    /**
     * Reorder to formats of BLAS.
     * The input format of MKLDNN is TNC, while that of BLAS is NTC.
     */
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

    val blasLSTM = nn.BiRecurrent[Float](nn.CAddTable()
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM(inputSize, hiddenSize))

    /**
     * biParams(0 - 2) and (3 - 5) are for the two directions respectively
     *
     * Gate order matching between MKLDNN LSTM and nn/LSTM:
     * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
     * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
     * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
     * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
     *
     * biParams(0) -> input weights
     * biParams(1) -> bias
     * biParams(2) -> hidden weights
     * biParams(3) -> input weights
     * biParams(4) -> bias
     * biParams(5) -> hidden weights
     */

    val biParams = blasLSTM.parameters()._1
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

    val blas_output = blasLSTM.forward(inputt).toTensor.transpose(1, 2)
    println("BLAS output LSTM Bi Sum \n" + blas_output)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(mkldnn_output).asInstanceOf[Tensor[Float]],
      blas_output) should be(true)
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

    val mkldnnLSTM = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, commonSize, commonSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter,
        initBias = initBias, layers = common_n_layers))
    mkldnnLSTM.evaluate()
    mkldnnLSTM.compile(InferencePhase)
    val output = mkldnnLSTM.forward(input)
    println("MKLDNN output LSTM Uni Multilayers Left2Right \n" + output)

    /**
     * Reorder to formats of BLAS.
     * The input format of MKLDNN is TNC, while that of BLAS is NTC.
     */
    var inputt = input.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(common_n_layers, commonSize, lstm_n_gates, commonSize))
      .transpose(2, 3).transpose(3, 4)
    initWeightIter = initWeightIter
      .resize(Array(common_n_layers, commonSize, lstm_n_gates, commonSize))
      .transpose(2, 3).transpose(3, 4)
    initBias = initBias.resize(Array(common_n_layers, lstm_n_gates, commonSize))

    /**
     * Gate order matching between MKLDNN LSTM and nn/LSTM:
     * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
     * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
     * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
     * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
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

    val blasLSTM = nn.Graph(nn_input, nn_lstm)

    val uniParams = blasLSTM.parameters()._1

    for(l <- 0 until common_n_layers) {
      initWeight0(l + 1) = initWeight0(l + 1).resizeAs(uniParams(3 * l))
      initBias0(l + 1) = initBias0(l + 1).resizeAs(uniParams(3 * l + 1))
      initWeightIter0(l + 1) = initWeightIter0(l + 1).resizeAs(uniParams(3 * l + 2))

      uniParams(3 * l).copy(initWeight0(l + 1))
      uniParams(3 * l + 1).copy(initBias0(l + 1))
      uniParams(3 * l + 2).copy(initWeightIter0(l + 1))
    }

    val blas_output = blasLSTM.forward(inputt).toTensor.transpose(1, 2)
    println("BLAS output LSTM Uni Multilayers Left2Right \n" + blas_output)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(output).asInstanceOf[Tensor[Float]],
      blas_output) should be(true)
  }

  "LSTM UnidirectionalTraining updateGradInput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    val gradOutputFormat = HeapData(Array(seqLength, batchSize, hiddenSize), Memory.Format.tnc)
    val input = Tensor(Array(seqLength, batchSize, inputSize)).rand(-1.0, 1.0)
    val gradOutput = Tensor(Array(seqLength, batchSize, hiddenSize)).rand(1.0, 1.0)

    var initWeight = Tensor[Float](
      Array(common_n_layers, 1,
        inputSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 1,
        hiddenSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 1,
        lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    val mkldnnLSTM = Sequential()
      .add(Input(inputFormat.shape, inputFormat.layout))
      .add(RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias))

    mkldnnLSTM.compile(TrainingPhase)
    mkldnnLSTM.forward(input)
    val mkldnn_gradInput = mkldnnLSTM.backward(input, gradOutput)
    println("MKLDNN LSTM Uni Left2Right gradInput\n" + mkldnn_gradInput)

    /**
      * Reorder to formats of BLAS.
      * The input format of MKLDNN is TNC, while that of BLAS is NTC.
      */
    var inputt = input.transpose(1, 2).clone()
    var gradOutputt = gradOutput.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(inputSize, lstm_n_gates, hiddenSize))
      .transpose(1, 2).transpose(2, 3)
    initWeightIter = initWeightIter.resize(Array(hiddenSize, lstm_n_gates, hiddenSize))
      .transpose(1, 2).transpose(2, 3)
    initBias = initBias.resize(Array(lstm_n_gates, hiddenSize))

    /**
      * Gate order matching between MKLDNN LSTM and nn/LSTM:
      * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
      * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
      * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
      * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
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

    val blasLSTM = nn.Recurrent().add(nn.LSTM(inputSize, hiddenSize))

    val uniParams = blasLSTM.parameters()._1
    initWeight0 = initWeight0.resizeAs(uniParams(0))
    initBias0 = initBias0.resizeAs(uniParams(1))
    initWeightIter0 = initWeightIter0.resizeAs(uniParams(2))

    uniParams(0).copy(initWeight0)
    uniParams(1).copy(initBias0)
    uniParams(2).copy(initWeightIter0)

    blasLSTM.forward(inputt).toTensor.transpose(1, 2)

    val blas_gradInput = blasLSTM.backward(inputt, gradOutputt).toTensor.transpose(1, 2)
    println("BLAS LSTM Uni Left2Right gradInput\n" + blas_gradInput)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(mkldnn_gradInput).asInstanceOf[Tensor[Float]],
      blas_gradInput) should be(true)
  }

  "LSTM BidirectionalSumTraining updateGradInput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val inputSize = 3
    val hiddenSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.BidirectionalSum

    val common_n_layers = 1
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.tnc)
    val gradOutputFormat = HeapData(Array(seqLength, batchSize, hiddenSize), Memory.Format.tnc)
    val input = Tensor(Array(seqLength, batchSize, inputSize)).rand()
    val gradOutput = Tensor(Array(seqLength, batchSize, hiddenSize)).rand(1.0, 1.0)

    var initWeight = Tensor[Float](
      Array(common_n_layers, 2,
        inputSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 2,
        hiddenSize, lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 2,
        lstm_n_gates, hiddenSize)).rand(-1.0, 1.0)

    val mkldnnLSTM = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, inputSize, hiddenSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias))

    mkldnnLSTM.compile(TrainingPhase)
    mkldnnLSTM.forward(input)
    val mkldnn_gradInput = mkldnnLSTM.backward(input, gradOutput)
    println("MKLDNN LSTM Bi Sum gradInput\n" + mkldnn_gradInput)

    /**
      * Reorder to formats of BLAS.
      * The input format of MKLDNN is TNC, while that of BLAS is NTC.
      */
    val inputt = input.transpose(1, 2).clone()
    var gradOutputt = gradOutput.transpose(1, 2).clone()
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

    val blasLSTM = nn.BiRecurrent[Float](nn.CAddTable()
      .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
      .add(nn.LSTM(inputSize, hiddenSize))

    /**
      * biParams(0 - 2) and (3 - 5) are for the two directions respectively
      *
      * Gate order matching between MKLDNN LSTM and nn/LSTM:
      * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
      * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
      * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
      * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
      *
      * biParams(0) -> input weights
      * biParams(1) -> bias
      * biParams(2) -> hidden weights
      * biParams(3) -> input weights
      * biParams(4) -> bias
      * biParams(5) -> hidden weights
      */

    val biParams = blasLSTM.parameters()._1
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

    blasLSTM.forward(inputt).toTensor.transpose(1, 2)

    val blas_gradInput = blasLSTM.backward(inputt, gradOutputt).toTensor.transpose(1, 2)
    println("BLAS LSTM Bi Sum gradInput \n" + blas_gradInput)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(mkldnn_gradInput).asInstanceOf[Tensor[Float]],
      blas_gradInput) should be(true)
  }

  "LSTM UnidirectionalInference Multilayers gradInput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 2
    val commonSize = 5

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 3
    val lstm_n_gates = 4

    val inputFormat = HeapData(Array(seqLength, batchSize, commonSize), Memory.Format.tnc)
    val gradOutputFormat = HeapData(Array(seqLength, batchSize, commonSize), Memory.Format.tnc)
    var input = Tensor(Array(seqLength, batchSize, commonSize)).rand()
    val gradOutput = Tensor(Array(seqLength, batchSize, commonSize)).rand(1.0, 1.0)

    var initWeight = Tensor[Float](
      Array(common_n_layers, 1,
        commonSize, lstm_n_gates, commonSize)).rand(-1.0, 1.0)

    var initWeightIter = Tensor[Float](
      Array(common_n_layers, 1,
        commonSize, lstm_n_gates, commonSize)).rand(-1.0, 1.0)

    var initBias = Tensor[Float](
      Array(common_n_layers, 1,
        lstm_n_gates, commonSize)).rand(-1.0, 1.0)

    val mkldnnLSTM = Sequential()
      .add(Input(input.size(), Memory.Format.tnc))
      .add(RNN(AlgKind.VanillaLstm, commonSize, commonSize, f, direction,
        initWeight = initWeight, initWeightIter = initWeightIter,
        initBias = initBias, layers = common_n_layers))

    mkldnnLSTM.compile(TrainingPhase)
    mkldnnLSTM.forward(input)
    val mkldnn_gradInput = mkldnnLSTM.backward(input, gradOutput)
    println("MKLDNN LSTM Uni Multilayers Left2Right gradInput\n" + mkldnn_gradInput)

    /**
      * Reorder to formats of BLAS.
      * The input format of MKLDNN is TNC, while that of BLAS is NTC.
      */
    var inputt = input.transpose(1, 2).clone()
    var gradOutputt = gradOutput.transpose(1, 2).clone()
    initWeight = initWeight.resize(Array(common_n_layers, commonSize, lstm_n_gates, commonSize))
      .transpose(2, 3).transpose(3, 4)
    initWeightIter = initWeightIter
      .resize(Array(common_n_layers, commonSize, lstm_n_gates, commonSize))
      .transpose(2, 3).transpose(3, 4)
    initBias = initBias.resize(Array(common_n_layers, lstm_n_gates, commonSize))

    /**
      * Gate order matching between MKLDNN LSTM and nn/LSTM:
      * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
      * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
      * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
      * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
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

    val blasLSTM = nn.Graph(nn_input, nn_lstm)

    val uniParams = blasLSTM.parameters()._1

    for(l <- 0 until common_n_layers) {
      initWeight0(l + 1) = initWeight0(l + 1).resizeAs(uniParams(3 * l))
      initBias0(l + 1) = initBias0(l + 1).resizeAs(uniParams(3 * l + 1))
      initWeightIter0(l + 1) = initWeightIter0(l + 1).resizeAs(uniParams(3 * l + 2))

      uniParams(3 * l).copy(initWeight0(l + 1))
      uniParams(3 * l + 1).copy(initBias0(l + 1))
      uniParams(3 * l + 2).copy(initWeightIter0(l + 1))
    }

    blasLSTM.forward(inputt).toTensor.transpose(1, 2)

    val blas_gradInput = blasLSTM.backward(inputt, gradOutputt).toTensor.transpose(1, 2)
    println("BLAS LSTM Uni Multilayers Left2Right gradInput\n" + blas_gradInput)
    println("==================================================================== \n\n\n")

    Equivalent.nearequals(Tools.dense(mkldnn_gradInput).asInstanceOf[Tensor[Float]],
      blas_gradInput) should be(true)
  }
}
