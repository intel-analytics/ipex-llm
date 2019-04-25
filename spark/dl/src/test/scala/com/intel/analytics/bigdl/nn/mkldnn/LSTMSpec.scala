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
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory, RNNCellFlags}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

class LSTMSpec extends FlatSpec with Matchers{
  "LSTM UnidirectionalLeft2Right updateOutput" should "work correctly" in {
    val seqLength = 3
    val batchSize = 1
    val inputSize = 2
    val hiddenSize = 2

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val common_n_layers = 1
    val lstm_n_gates = 4
    val lstm_n_states = 2

    val inputFormat = HeapData(Array(seqLength, batchSize, inputSize), Memory.Format.any)

    var input = Tensor(Array(seqLength, batchSize, inputSize))
    for (i <- 1 to seqLength; j <- 1 to batchSize; k <- 1 to inputSize)
      input.setValue(i, j, k, 2f)

    // val input = Tensor(Array(seqLength, batchSize, inputSize)).rand()
    // println("input")
    // println(input)

    var initWeight = Tensor[Float](Array(common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to inputSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeight.setValue(a, b, i, j, k, 0.5f)
    /* val initWeight = Tensor[Float](common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeight")
    // println(initWeight)

    var initWeightIter = Tensor[Float](Array(common_n_layers, 1,
      hiddenSize, lstm_n_gates, hiddenSize))
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to hiddenSize; j <- 1 to lstm_n_gates; k <- 1 to hiddenSize)
      initWeightIter.setValue(a, b, i, j, k, 0.5f)
    /* val initWeightIter = Tensor[Float](common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize)
      .rand() */
    // println("initWeightIter")
    // println(initWeightIter)

    var initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize)
    for (a <- 1 to common_n_layers; b <- 1 to 1;
         i <- 1 to lstm_n_gates; j <- 1 to hiddenSize)
      initBias.setValue(a, b, i, j, 1f)
    // val initBias = Tensor[Float](common_n_layers, 1, lstm_n_gates, hiddenSize).rand()
    // println("initBias")
    // println(initBias)

    val lstm1 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    /* val lstm = LSTM(inputSize, hiddenSize, f, flags, alpha, clipping,
         direction) */
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initMemoryDescs(Array(inputFormat))
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output1 = lstm1.forward(input)
    println("DNN output Left2Right\n" + output1)

    direction = Direction.UnidirectionalRight2Left
    val lstm2 = LSTM(inputSize, hiddenSize, f, direction,
      initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
    /* val lstm = LSTM(inputSize, hiddenSize, f, flags, alpha, clipping,
         direction) */
    lstm2.setRuntime(new MklDnnRuntime)
    lstm2.initMemoryDescs(Array(inputFormat))
    lstm2.initFwdPrimitives(Array(inputFormat), InferencePhase)
    val output2 = lstm2.forward(input)
    println("DNN output Right2Left\n" + output2)


    // input = input.resize(Array(batchSize, seqLength, inputSize))

    /*
    val nn_model = nn.Sequential().add(
      nn.Recurrent().add(nn.LSTM2(inputSize, hiddenSize))
    )

    val (nn_weight, nn_grad) = nn_model.getParameters()
    println(nn_weight)

    val nn_output = nn_model.forward(input).toTensor
    println(nn_output)
    */
  }
}
